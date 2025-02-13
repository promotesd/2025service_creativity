import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel


import os
import torch
from typing import List
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from peft import PeftModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

class LineChunkSplitter:
    def __init__(self, lines_per_chunk=5):
        self.lines_per_chunk = lines_per_chunk
    def split_text(self, text: str) -> List[str]:
        lines = text.splitlines()
        chunks, cache = [], []
        for i, line in enumerate(lines):
            cache.append(line)
            if (i+1) % self.lines_per_chunk == 0:
                chunks.append("\n".join(cache))
                cache = []
        if cache:
            chunks.append("\n".join(cache))
        return chunks
    def split_documents(self, docs: List[Document]) -> List[Document]:
        new_docs = []
        for doc in docs:
            splitted = self.split_text(doc.page_content)
            for c in splitted:
                new_docs.append(Document(page_content=c, metadata=doc.metadata))
        return new_docs

class LoraT5Summarizer:
    def __init__(self, base_model: str, lora_model_dir: str, device="cuda"):
        self.tokenizer = T5Tokenizer.from_pretrained(base_model)
        self.base_t5 = T5ForConditionalGeneration.from_pretrained(base_model)
        self.model = PeftModel.from_pretrained(self.base_t5, lora_model_dir)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def summarize(self, text: str, max_length=80) -> str:
        if not text.strip():
            return ""
        prompt = f"概括: {text}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=4,
                do_sample=False,
                no_repeat_ngram_size=2
            )
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()

def main():
    file_path = "/root/autodl-tmp/code/2025service_creativity/process_dataset/mathNER/train.txt"
    persist_dir = "./chroma_summaries_lora_db"
    os.makedirs(persist_dir, exist_ok=True)

    # 1) 加载原始文本
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()
    print(f"原始文档数: {len(docs)}")

    # 2) 行级切分
    splitter = LineChunkSplitter(lines_per_chunk=5)
    splitted_docs = splitter.split_documents(docs)
    print(f"拆分后文档数: {len(splitted_docs)}")

    # 3) 加载LoRA微调后的T5做摘要
    base_model_name = "uer/t5-base-chinese-cluecorpussmall"
    lora_dir = "./chinese_t5_lora"  # 你在第一步训练生成的目录
    summarizer = LoraT5Summarizer(base_model_name, lora_dir, device="cuda")

    # 4) 对每段文本做摘要
    summary_docs = []
    for doc in splitted_docs:
        original_text = doc.page_content
        summary_text = summarizer.summarize(original_text, max_length=80)
        summary_docs.append(Document(
            page_content=summary_text,
            metadata={"original_text": original_text}
        ))
    print(f"生成摘要文档数: {len(summary_docs)}")

    # 5) 中文 Embedding: e.g. text2vec
    embedding_model = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")

    # 6) 存入Chroma向量库
    vectorstore = Chroma.from_documents(
        documents=summary_docs,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    print(f"Chroma数据库已保存到: {persist_dir}")

    # 7) 测试检索
    query = "函数的定义是什么？"
    results = vectorstore.similarity_search(query, k=2)
    print(f"\n查询: {query}")
    for i, r in enumerate(results):
        print(f"【Top {i+1}】摘要: {r.page_content}")
        print(f"原文: {r.metadata['original_text']}")

if __name__ == "__main__":
    main()
