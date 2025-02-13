#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_inference_advanced.py

实现一个更复杂的推理流程，集成:
- Qwen2-Math + LoRA 加载 & 合并
- 多轮对话 & 流式输出
- 定理溯源(若有Google Search API，否则Chroma数据库)
- OCR解析图片后再提交大模型
"""

import os
import sys
import time
import torch
import requests

import pytesseract
from PIL import Image

# 如果需要 langchain/chroma
# pip install langchain chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
# Google Search
# pip install google-search-results
try:
    from langchain.utilities import GoogleSearchAPIWrapper
    HAS_GOOGLE = True
except:
    HAS_GOOGLE = False

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel

###########################################
# 1. 全局配置: paths, keys
###########################################
BASE_MODEL_PATH = "/root/autodl-tmp/model/Qwen2-math7B"
LORA_WEIGHT_PATH = "/root/autodl-tmp/code/2025service_creativity/Qwen-math_train/loraSFT/run_20250213_163025/checkpoint-40"
CHROMA_DB_DIR = "/root/autodl-tmp/code/2025service_creativity/Qwen-math_train/chroma_db"  # 你的Chroma数据库目录
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")  # 如果没有则为空
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID", "")    # 同理

###########################################
# 2. 全局构造: Qwen model + LoRA + Tokenizer
###########################################
def load_model():
    print("[Loading base Qwen2-Math7B model...]")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype="auto",
        device_map="auto"
    )
    print("[Loading LoRA weights...]")
    peft_model = PeftModel.from_pretrained(base_model, LORA_WEIGHT_PATH)
    # merge
    model = peft_model.merge_and_unload()
    model.eval()
    return tokenizer, model

###########################################
# 3. 定理溯源: Google or local Chroma
###########################################
def theorem_search(query: str) -> str:
    """
    如果google key可用 => google搜索 -> 将搜索top结果
    否则 => local chroma vectorstore
    返回文本信息
    """
    # 1) google
    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        print("[Info] Using Google Search API for theorem retrieval.")
        from langchain.utilities import GoogleSearchAPIWrapper
        search_tool = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)
        results = search_tool.results(query, num=3)
        # 合并摘要
        combined_text = ""
        for r in results:
            title = r.get("title","")
            snippet = r.get("snippet","")
            link = r.get("link","")
            combined_text += f"Title: {title}\nSnippet: {snippet}\nURL: {link}\n\n"
        return combined_text
    else:
        print("[Info] No Google API key found, fallback to local Chroma search.")
        # local chroma
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vectorstore = Chroma(collection_name="mytheorem", embedding_function=embedding, persist_directory=CHROMA_DB_DIR)
        results = vectorstore.similarity_search(query, k=3)
        combined_text = ""
        for doc in results:
            combined_text += f"[LocalDoc] {doc.page_content}\n"
        return combined_text

###########################################
# 4. 流式输出 + Qwen 
###########################################
def stream_generate(model, tokenizer, prompt: str):
    """
    用 huggingface 的 TextIteratorStreamer 实现流式输出
    """
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs=input_ids.input_ids,
        max_new_tokens=25000,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        streamer=streamer
    )

    # 异步生成
    import threading
    def gen_func():
        model.generate(**generation_kwargs)

    thread = threading.Thread(target=gen_func)
    thread.start()

    print("Assistant: ", end="", flush=True)
    for new_token in streamer:
        print(new_token, end="", flush=True)
    print("\n---Done---")
    thread.join()

###########################################
# 5. 主流程: OCR + Theorem search + Chat
###########################################
def main():
    tokenizer, model = load_model()

    while True:
        user_input = input("\n[User] ").strip()
        if not user_input:
            print("结束对话.")
            break
        
        # check if it's an image: prefix => "image: /path/or/url"
        if user_input.startswith("image:"):
            image_path = user_input[len("image:"):].strip()
            # OCR
            try:
                ocr_text = do_ocr(image_path)
                print(f"[OCR result] {ocr_text}")
                user_input = ocr_text
            except Exception as e:
                print(f"[Error] OCR failed: {e}")
                user_input = "(OCR失败)"

        # check theorem search
        # 你可判断user中是否出现"定理:"之类. 这里直接user问 => we do search
        # for demonstration:
        print("[Info] Doing theorem search ...")
        retrieved_text = theorem_search(user_input)
        # 在prompt开头插入
        system_msg = "你是一个热于助人的数学助手，解题时，如果需要请提供思维图。LATEX格式输出\n"
        if retrieved_text.strip():
            system_msg += f"\n[定理溯源结果]:\n{retrieved_text}\n"
        
        # 构建实际prompt
        # Qwen style: <system> system_msg</system> <user> user_input </user> <assistant>...
        messages = [
            {"role":"system","content":system_msg},
            {"role":"user","content":user_input}
        ]
        text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 流式输出
        stream_generate(model, tokenizer, text_prompt)

def do_ocr(image_path):
    """
    简易OCR. 需: pip install pytesseract pillow
    可能还需 tesseract可执行文件
    """
    # 如果是url, 先下载
    if image_path.startswith("http"):
        import requests
        import tempfile
        resp = requests.get(image_path)
        if resp.status_code==200:
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmpf.write(resp.content)
            tmpf.close()
            image_path = tmpf.name
        else:
            raise ValueError("下载图片失败.")
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="chi_sim+eng")  # or others
    return text

if __name__=="__main__":
    main()
