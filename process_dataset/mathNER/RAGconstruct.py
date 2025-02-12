import os
from typing import List
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline  # 如果需要调用HF模型也可用
from transformers import pipeline

# ==================== 1) 行级拆分器 ====================
class LineChunkSplitter:
    """
    将文本按行读取，并每N行合并成一个chunk，作为一条Document。
    """
    def __init__(self, lines_per_chunk: int = 5):
        self.lines_per_chunk = lines_per_chunk

    def split_text(self, text: str) -> List[str]:
        lines = text.splitlines()
        chunks = []
        current_lines = []
        for i, line in enumerate(lines):
            current_lines.append(line)
            if (i + 1) % self.lines_per_chunk == 0:
                chunk_text = "\n".join(current_lines)
                chunks.append(chunk_text)
                current_lines = []
        if current_lines:
            chunk_text = "\n".join(current_lines)
            chunks.append(chunk_text)
        return chunks

    def split_documents(self, docs: List[Document]) -> List[Document]:
        new_docs = []
        for doc in docs:
            chunks = self.split_text(doc.page_content)
            for chunk in chunks:
                new_docs.append(Document(page_content=chunk, metadata=doc.metadata))
        return new_docs

# ==================== 2) 摘要器 ====================
def summarize_text(text: str, summary_pipeline, max_len=60) -> str:
    """
    调用 HF 的文本摘要 pipeline，对 chunk 做简短摘要。
    max_len: 控制摘要输出的最大token数
    """
    if not text.strip():
        return ""

    # pipeline会返回一个list of dict
    # e.g. [{'summary_text': '...'}]
    result = summary_pipeline(text, max_length=max_len, truncation=True)[0]
    summary = result["summary_text"]
    return summary

# ==================== 3) 主流程 ====================
def main():
    # (A) 基本配置
    file_path = "/root/autodl-tmp/code/2025service_creativity/process_dataset/mathNER/train.txt"
    persist_dir = "./chroma_summaries_db"

    # (B) 加载原始文本 Document
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()
    print(f"原始文档数: {len(docs)}")  # 多数情况下是1

    # (C) 行级拆分
    splitter = LineChunkSplitter(lines_per_chunk=5)
    split_docs = splitter.split_documents(docs)
    print(f"拆分后文档数: {len(split_docs)}")

    # (D) 初始化一个Hugging Face的摘要pipeline
    #   示例使用 facebook/bart-large-cnn 或 t5-small 等可做摘要的模型
    #   当然也可以用中文专用的，比如 'IDEA-CCNL/Wenzhong2.0-GPT-0.3B-Summary'
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")

    # (E) 对每个 chunk 做摘要，并构造新的 "摘要文档"
    summary_docs = []
    for doc in split_docs:
        chunk_text = doc.page_content
        summary_text = summarize_text(chunk_text, summarizer, max_len=60)
        # 我们把 "summary_text" 当做实际索引向量的文本
        # 同时把原文本写入 metadata["original_text"]
        summary_docs.append(
            Document(
                page_content=summary_text, 
                metadata={
                    "original_text": chunk_text
                }
            )
        )

    print(f"生成摘要文档数: {len(summary_docs)}")

    # (F) 初始化 Embedding
    # 这里随便用了 huggingface all-MiniLM-L6-v2
    # 也可以换成 Qwen Embedding(若有对应embedding接口)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # (G) 写入 Chroma
    vectorstore = Chroma.from_documents(
        documents=summary_docs,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    print(f"已构建并保存向量库到: {persist_dir}")

    # (H) 测试检索
    query = "函数的概念"
    results = vectorstore.similarity_search(query, k=2)

    print(f"\n查询: {query}")
    for i, r in enumerate(results):
        print(f"=== Top {i+1} 命中摘要 ===")
        print("【摘要】:", r.page_content)
        print("【原文】:", r.metadata["original_text"])
        print("------")

if __name__ == "__main__":
    main()
