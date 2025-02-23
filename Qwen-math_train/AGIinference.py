#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AGIinference.py

功能更新：
1. 使用新版Chroma配置 (避免Deprecated Error)，数据库路径指定到:
   "/root/autodl-tmp/code/2025service_creativity/Qwen-math_train/AGIchroma"

2. 用户输入的问题 & 模型输出的答案写入Chroma数据库，以便下次检索相似问题。

3. 可选: 用 pylatexenc 将大模型输出中的 LaTeX 转成纯文本(若需要命令行可读),
   也可直接保留原LaTeX字符串。

4. 其余功能与原先类似：
   - 多轮对话(存到 conversation_log.json)
   - 流式输出 + Ctrl+C 中断
   - OCR 处理 (对复杂公式有限支持, 建议使用更专业的公式识别工具)
   - 如果有 google search API / Chroma DB，可做定理检索
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

# 如果需要 langchain/chroma
# pip install langchain chromadb google-search-results
# + pylatexenc(可选)
try:
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.utilities import GoogleSearchAPIWrapper
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

###########################################
# 0. 全局配置
###########################################
BASE_MODEL_PATH = "/root/autodl-tmp/model/Qwen2-math7B"
LORA_WEIGHT_PATH = "/root/autodl-tmp/code/2025service_creativity/Qwen-math_train/loraSFT/run_20250213_171645/checkpoint-100000"

CONV_LOG_FILE = "conversation_log.json"  # 用于持久化多轮对话

# Chroma数据库路径
CHROMA_DB_DIR  = "/root/autodl-tmp/code/2025service_creativity/Qwen-math_train/AGIchroma"

# 如果有 Google Search API：
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID  = os.environ.get("GOOGLE_CSE_ID", "")

# 初始系统提示
SYSTEM_PROMPT = "你是一位热心且专业的数学助手，能进行图文OCR、多轮对话与定理检索。回答时请适度使用LaTeX格式。"

###########################################
# 1. 加载Qwen + LoRA 并合并
###########################################
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
    model = peft_model.merge_and_unload()  # merge weights
    model.eval()
    return tokenizer, model

###########################################
# 2. 流式输出
###########################################
class StopGeneration(Exception):
    """用于中断生成的自定义异常"""
    pass

def stream_generate(model, tokenizer, prompt: str, max_new_tokens=1024, temperature=0.7, top_p=0.9):
    """
    用 huggingface TextIteratorStreamer 实现流式输出，并支持 Ctrl+C 中断
    """
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    gen_kwargs = dict(
        **input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
    )

    def generate_func():
        model.generate(**gen_kwargs)

    thread = Thread(target=generate_func)
    thread.start()

    print("Assistant: ", end="", flush=True)
    try:
        for new_token in streamer:
            print(new_token, end="", flush=True)
    except KeyboardInterrupt:
        # 用户手动 Ctrl+C
        print("\n[User Interrupted Generation!]")
    finally:
        thread.join()
    print("\n---Done---")

###########################################
# 3. 定理或知识检索 (chroma new usage)
###########################################
def theorem_search(query):
    if not HAS_LANGCHAIN:
        return ""
    import chromadb
    from chromadb.config import Settings

    # ⚠ 改为 chroma_db_impl="duckdb+parquet"
    client = chromadb.Client(Settings(
        persist_directory=CHROMA_DB_DIR,
        chroma_db_impl="duckdb+parquet",
        anonymized_telemetry=False
    ))
    # get or create
    coll = client.get_or_create_collection("theorem_docs")

    # 如果有Google Key => google搜索
    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        print("[Info] Using Google Search API for retrieval.")
        search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)
        results = search.results(query, num=3)
        combined_text = ""
        for r in results:
            title = r.get("title","")
            snippet = r.get("snippet","")
            link = r.get("link","")
            combined_text += f"Title: {title}\nSnippet: {snippet}\nURL: {link}\n\n"
        return combined_text
    else:
        print("[Info] No Google API key, fallback to local theorem Chroma DB.")
        # 用 langchain's Chroma VectorStore
        from langchain.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        from langchain.vectorstores import Chroma as LGChroma

        vectordb = LGChroma(collection_name="theorem_docs",
                            embedding_function=embeddings,
                            persist_directory=CHROMA_DB_DIR)
        docs = vectordb.similarity_search(query, k=3)
        combined_text = "\n".join([f"[LocalDoc] {d.page_content}" for d in docs])
        return combined_text

###########################################
# 4. OCR
###########################################
def do_ocr(image_path: str):
    """传入本地路径或 http(s) url, 返回OCR文本"""
    if image_path.startswith("http"):
        resp = requests.get(image_path)
        if resp.status_code == 200:
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmpf.write(resp.content)
            tmpf.close()
            image_path = tmpf.name
        else:
            raise RuntimeError("下载图片失败，无法OCR。")
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="chi_sim+eng")
    return text.strip()

###########################################
# 5. 对话管理(多轮) + Chroma存储问答
###########################################
def load_conversation():
    if not os.path.exists(CONV_LOG_FILE):
        return []
    try:
        with open(CONV_LOG_FILE, "r", encoding="utf-8") as f:
            conv_data = json.load(f)
            if isinstance(conv_data, list):
                return conv_data
            else:
                return []
    except:
        return []

def save_conversation(conv_messages):
    with open(CONV_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(conv_messages, f, ensure_ascii=False, indent=2)

def build_prompt_from_messages(conv_messages):
    final_text = f"<system>\n{SYSTEM_PROMPT}\n</system>\n"
    for msg in conv_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            final_text += f"<system>\n{content}\n</system>\n"
        elif role == "user":
            final_text += f"<user>\n{content}\n</user>\n"
        else:
            final_text += f"<assistant>\n{content}\n</assistant>\n"
    final_text += "<assistant>\n"
    return final_text

def store_qa_in_chroma(question: str, answer: str):
    import chromadb
    from chromadb.config import Settings

    # ⚠ 同样要用新的设置
    client = chromadb.Client(Settings(
        persist_directory=CHROMA_DB_DIR,
        chroma_db_impl="duckdb+parquet",
        anonymized_telemetry=False
    ))
    collection = client.get_or_create_collection("qa_pairs")

    doc_id = f"qa-{int(time.time())}"
    merged_text = f"Q: {question}\nA: {answer}"
    collection.add(documents=[merged_text], metadatas=[{"question": question}], ids=[doc_id])
    print(f"[Chroma] 已存Q&A => doc_id={doc_id}")

###########################################
# 6. 主函数
###########################################
def main():
    tokenizer, model = load_model_and_tokenizer()

    conv_messages = load_conversation()
    print(f"已加载历史对话，共 {len(conv_messages)} 条消息。\n若要重新开始，可删除 {CONV_LOG_FILE} 文件。")

    while True:
        user_input = input("\n[User] ").strip()
        if not user_input:
            print("结束对话.")
            break
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("用户结束对话.")
            break

        # 如果是图片
        if user_input.startswith("image:"):
            image_path = user_input[len("image:"):].strip()
            try:
                ocr_res = do_ocr(image_path)
                print("[OCR result] ", ocr_res)
                user_input = f"(OCR结果)\n{ocr_res}"
            except Exception as e:
                print(f"[Error in OCR]: {e}")
                user_input = "(OCR失败)"

        # 检索
        retrieval_text = theorem_search(user_input)
        if retrieval_text:
            user_input += f"\n\n[检索到的参考信息]:\n{retrieval_text}"

        # 追加到对话
        conv_messages.append({"role": "user", "content": user_input})
        prompt_text = build_prompt_from_messages(conv_messages)

        print("\n[Assistant 正在思考]...\n")
        stream_generate(model, tokenizer, prompt_text, max_new_tokens=512, temperature=0.6, top_p=0.9)

        # 让用户手动复制回答
        print("是否保存这轮助手回答到对话历史&Chroma？(y/n)")
        ans = input("> ").strip().lower()
        if ans in ["y","yes"]:
            print("请把上面助手回答(含LaTeX)复制粘贴到这里(回车结束):")
            assistant_text = []
            while True:
                line = input()
                if not line:
                    break
                assistant_text.append(line)
            final_answer = "\n".join(assistant_text).strip()
            if final_answer:
                # 可选做LaTeX解析->纯文本
                # text_only = LatexNodes2Text().latex_to_text(final_answer)
                conv_messages.append({"role":"assistant", "content": final_answer})
                save_conversation(conv_messages)
                print("[已保存对话历史]\n")

                # 存入Chroma
                store_qa_in_chroma(user_input, final_answer)
            else:
                print("[未输入任何文本，跳过保存]\n")
        else:
            print("[跳过保存]\n")

    print("对话结束，再见。")

if __name__ == "__main__":
    main()
