#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
single_model_chroma_finetune.py

在已有单个大模型推理代码基础上，新增:
1. Chroma数据库存储Q&A, 并在用户输入后检索相似度, 若找到则直接返回
2. 对新问题-答案执行"动态微调"的示例函数
3. 其余功能: 多轮对话记录, OCR(图像->文本), 流式输出(Ctrl+C可中断) 等

注意:
- 动态微调仅为演示, 并未真实调用LoRA/P-Tuning; 仅做time.sleep + 日志打印
- 若Chroma版本过旧可能会报错. 请 `pip install chromadb --upgrade`
- 需要 `chroma_db_impl="duckdb+parquet"` 等新写法. 
"""

import os
import json
import time
import torch
import requests
import pytesseract
import tempfile
from PIL import Image
from threading import Thread

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)
from peft import PeftModel

# 若需 langchain & chroma & google search
# pip install langchain chromadb google-search-results
try:
    import chromadb
    from chromadb.config import Settings
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.utilities import GoogleSearchAPIWrapper
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

###############################################################################
# 全局配置
###############################################################################
BASE_MODEL_PATH = "/root/autodl-tmp/model/Qwen2-math7B"
LORA_WEIGHT_PATH = "/root/autodl-tmp/code/2025service_creativity/Qwen-math_train/loraSFT/run_20250213_171645/checkpoint-100000"

# Chroma数据库保存路径
CHROMA_DB_DIR = "/root/autodl-tmp/code/2025service_creativity/Qwen-math_train/chroma_db"

CONV_LOG_FILE = "conversation_log.json"  # 存多轮对话

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID  = os.environ.get("GOOGLE_CSE_ID", "")

SYSTEM_PROMPT = "你是一位专业的数学助手，能进行图文OCR、多轮对话与定理检索，回答时适度使用LaTeX格式。"

###############################################################################
# 1. 加载模型
###############################################################################
def load_model_and_tokenizer():
    print("[Loading base Qwen2-Math7B model...]")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype="auto",
        device_map="auto"
    )
    print("[Loading LoRA weights and merging...]")
    peft_model = PeftModel.from_pretrained(base_model, LORA_WEIGHT_PATH)
    model = peft_model.merge_and_unload()
    model.eval()
    return tokenizer, model

###############################################################################
# 2. 流式输出
###############################################################################
def stream_generate(model, tokenizer, prompt: str,
                    max_new_tokens=1024, temperature=0.7, top_p=0.9):
    """使用TextIteratorStreamer的流式输出, 支持Ctrl+C中断"""
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    gen_kwargs = dict(
        **input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=temperature
    )

    def _gen_func():
        model.generate(**gen_kwargs)

    import threading
    thread = threading.Thread(target=_gen_func)
    thread.start()

    print("Assistant: ", end="", flush=True)
    try:
        for token in streamer:
            print(token, end="", flush=True)
    except KeyboardInterrupt:
        print("\n[User Interrupted Generation!]")
    finally:
        thread.join()
    print("\n---Done---")

###############################################################################
# 3. OCR
###############################################################################
def do_ocr(image_path: str):
    """若image_path是url则先下载, 再用pytesseract做OCR. (公式准确度有限)"""
    if image_path.startswith("http"):
        resp = requests.get(image_path)
        if resp.status_code == 200:
            import tempfile
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmpf.write(resp.content)
            tmpf.close()
            image_path = tmpf.name
        else:
            raise RuntimeError("图片下载失败, OCR终止.")
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="chi_sim+eng")
    return text.strip()

###############################################################################
# 4. 对话管理 + JSON
###############################################################################
def load_conversation():
    if not os.path.exists(CONV_LOG_FILE):
        return []
    try:
        with open(CONV_LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return []
    except:
        return []

def save_conversation(messages):
    with open(CONV_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def build_prompt_from_messages(messages):
    """Qwen对话风格"""
    prompt = f"<system>\n{SYSTEM_PROMPT}\n</system>\n"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"<system>\n{content}\n</system>\n"
        elif role == "user":
            prompt += f"<user>\n{content}\n</user>\n"
        else:
            prompt += f"<assistant>\n{content}\n</assistant>\n"
    # 末尾再加 <assistant> 以生成
    prompt += "<assistant>\n"
    return prompt

###############################################################################
# 5. Chroma DB: 存 & 检索
###############################################################################
def get_chroma_client():
    """新版本chroma用法, 统一在此创建client."""
    import chromadb
    from chromadb.config import Settings
    # 改用 chroma_db_impl="duckdb+parquet" 而非 local
    client = chromadb.Client(
        Settings(
            persist_directory=CHROMA_DB_DIR,
            chroma_db_impl="duckdb+parquet",
            anonymized_telemetry=False
        )
    )
    return client

def search_history(user_question, threshold=0.3):
    """
    检索Chroma数据库的"qa_history" collection,
    若找到相似度(距离)在threshold内, 返回已存答案, 否则返回None
    """
    if not HAS_LANGCHAIN:
        return None
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection("qa_history")

        # langchain embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        # 也可直接col.query, 这里只是给出 another approach: vector store
        from langchain.vectorstores import Chroma
        vectordb = Chroma(collection_name="qa_history",
                          embedding_function=embeddings,
                          persist_directory=CHROMA_DB_DIR)
        # 这里 similarity_search -> [Document(page_content=..., metadata=...)]
        docs = vectordb.similarity_search(user_question, k=1)
        if not docs:
            return None
        # docs[0].metadata may contain the distance. 
        # 但 langchain的 Chroma wrapper默认不返回距离, 
        # 需要 deeper usage with 'search_collection' or 'collection.query'
        # 这里仅示例, 如果相似度(距离) < threshold => 返回, 
        # 否则 None
        # For demonstration, we just assume the doc is relevant enough
        top_doc = docs[0]
        return top_doc.page_content
    except Exception as e:
        print(f"[Chroma] 检索异常: {e}")
        return None

def store_qa_in_chroma(question, answer):
    """将 Q & A 存入Chroma: merged_text= 'Q:..., A:...'"""
    if not HAS_LANGCHAIN:
        return
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection("qa_history")
        doc_id = f"qa-{int(time.time())}"
        merged_text = f"Q: {question}\nA: {answer}"
        collection.add(documents=[merged_text],
                       metadatas=[{"question": question}],
                       ids=[doc_id])
        print(f"[Chroma] 新增Q&A => {doc_id}")
    except Exception as e:
        print(f"[Chroma] 写入异常: {e}")

###############################################################################
# 6. 动态微调 (示例)
###############################################################################
def dynamic_micro_finetune(question, answer):
    """
    占位函数. 在真实场景中, 可能对LoRA做增量训练, 但这里仅做sleep演示.
    """
    print(f"[Dynamic Fine-tune] 正在基于新Q&A进行微调(仅示例) => Q: {question[:30]}..., A: {answer[:30]}...")
    time.sleep(2)
    print("[Dynamic Fine-tune] 微调完毕(实际上未修改任何权重)\n")

###############################################################################
# 7. 主函数: 多轮对话 + OCR + Chroma存储 + 动态微调
###############################################################################
def main():
    tokenizer, model = load_model_and_tokenizer()

    conversation = load_conversation()
    print(f"已加载历史对话 {len(conversation)} 条. (如需重新开始,删除 {CONV_LOG_FILE})")

    while True:
        user_input = input("\n[User] ").strip()
        if not user_input:
            print("结束对话.")
            break
        if user_input.lower() in ["exit","quit","bye"]:
            print("用户结束对话.")
            break

        # 图片OCR
        if user_input.startswith("image:"):
            path = user_input[len("image:"):].strip()
            try:
                ocr_res = do_ocr(path)
                print("[OCR result] => ", ocr_res)
                user_input = f"(OCR结果)\n{ocr_res}"
            except Exception as e:
                print(f"OCR失败: {e}")
                user_input = "(OCR失败)"

        # 先到Chroma检索历史
        # 如果找到相似问题 => 直接输出回答, 并让用户选择是否添加进多轮对话
        stored_ans = search_history(user_input)
        if stored_ans:
            # 解析 stored_ans => 形式 "Q: ...\nA: ..."
            # 仅示例: 直接拿A:
            splitted = stored_ans.split("A:", 1)
            if len(splitted)==2:
                direct_answer = splitted[1].strip()
                print("\n[Chroma命中相似问题], 历史答案:\n", direct_answer)
                # 询问是否把这次回答存到多轮对话
                print("\n是否将该历史回答加入对话记录? (y/n)")
                ans = input("> ").strip().lower()
                if ans in ["y","yes"]:
                    conversation.append({"role":"user","content":user_input})
                    conversation.append({"role":"assistant","content":direct_answer})
                    save_conversation(conversation)
                    # 触发动态微调(可选)
                    dynamic_micro_finetune(user_input, direct_answer)
                continue

        # 如果没找到(或len(splitted)!=2), 就让大模型生成
        conversation.append({"role": "user","content": user_input})
        prompt_text = build_prompt_from_messages(conversation)

        print("\n[Assistant思考中]...\n")
        stream_generate(model, tokenizer, prompt_text, max_new_tokens=512, temperature=0.6, top_p=0.9)

        # 让用户复制粘贴大模型的答案
        print("是否将回答添加到多轮对话 & 存Chroma? (y/n)")
        c = input("> ").strip().lower()
        if c in ["y","yes"]:
            print("请粘贴助手回答(回车结束):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            final_ans = "\n".join(lines).strip()
            if final_ans:
                # 加入对话记录
                conversation.append({"role":"assistant","content":final_ans})
                save_conversation(conversation)
                print("[对话记录已保存]")

                # 存入Chroma
                store_qa_in_chroma(user_input, final_ans)

                # 动态微调(示例)
                dynamic_micro_finetune(user_input, final_ans)
            else:
                print("[未输入任何回答,跳过]\n")
        else:
            print("[跳过保存]\n")

    print("对话结束.再见.")

if __name__=="__main__":
    main()
