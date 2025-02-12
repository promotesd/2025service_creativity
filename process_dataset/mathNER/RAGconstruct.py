import os
import torch
import torch.nn as nn
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from typing import List

# ========== 1. 行级拆分器 ==========
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
                # 满N行，合并
                chunk_text = "\n".join(current_lines)
                chunks.append(chunk_text)
                current_lines = []
        # 处理剩余不足N行的
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

# ========== 2. 一个简易的 MLP 降维器 ==========
class MLPCompressor(nn.Module):
    """
    示例：把768维向量 -> 256维
    可以根据需要做更复杂结构或不同 hidden_dim。
    """
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=256):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: shape (batch, input_dim)
        return: shape (batch, output_dim)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


# ========== 3. 自定义的 Qwen Math Embedding (带 MLP 压缩) ==========
class QwenMathEmbedding(Embeddings):
    """
    假设你有一个Qwen-Math模型可输出768维向量，
    然后再用MLPCompressor降维到256(仅作示例)。
    """

    def __init__(self, qwen_model_path: str, mlp_path: str, device: str = "cuda"):
        """
        qwen_model_path: Qwen Embedding模型的路径或名称
        mlp_path: 事先训练好的MLP权重文件
        device: cuda 或 cpu
        """
        super().__init__()
        self.device = device
        # 这里你需要替换成真实的 Qwen Embedding 初始化逻辑:
        # self.qwen_tokenizer = ...
        # self.qwen_model = ...
        # 例如 self.qwen_model = QwenForSentenceEmbedding.from_pretrained(...).to(device)

        # 为了示例，我们假设 output_dim=768
        self.output_dim = 768

        # 初始化 MLP
        self.mlp = MLPCompressor(input_dim=768, hidden_dim=256, output_dim=256).to(device)
        # 加载 MLP 已训练好的权重
        self.mlp.load_state_dict(torch.load(mlp_path))

        self.mlp.eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        对多段文本做 embedding，然后用 MLP 降维
        返回 2D list: [[x1, x2, ..., x256], ...]
        """
        embeddings = []
        for txt in texts:
            emb = self._embed_one(txt)
            embeddings.append(emb)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        对单条查询文本做 embedding
        """
        emb = self._embed_one(text)
        return emb

    def _embed_one(self, text: str) -> List[float]:
        # 这里是你真实的 Qwen embedding过程 (伪代码)
        # inputs = self.qwen_tokenizer(text, return_tensors="pt").to(self.device)
        # with torch.no_grad():
        #     outputs = self.qwen_model(**inputs)
        #     vec_768 = outputs["sentence_embedding"]  # shape (1,768)

        # 为了演示，这里用随机向量代替
        vec_768 = torch.randn(1, self.output_dim).to(self.device)

        # 用 MLP 压缩
        with torch.no_grad():
            vec_256 = self.mlp(vec_768)  # shape (1,256)
        return vec_256.squeeze(0).tolist()  # 转成 python list

# ========== 4. 主流程: 加载 -> 行级切分 -> Embedding -> Chroma数据库 ==========
def main():
    # 1) 文件路径
    file_path = "/root/autodl-tmp/code/2025service_creativity/process_dataset/mathNER/train.txt"
    persist_dir = "./mathner_chroma_db_compressed"  # 向量数据库存放目录
    mlp_weight_path = "./mlp_compressor.pt"  # 事先训练好的 MLP 权重（示例）
    qwen_model_path = "./qwen_math_embedding"       # 你的 Qwen Embedding 模型路径

    # 2) 加载文本 (单个Document)
    from langchain.document_loaders import TextLoader
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()

    print(f"原始文档数: {len(docs)}")

    # 3) 行级拆分
    line_splitter = LineChunkSplitter(lines_per_chunk=5)  # 每5行合并成一段
    split_docs = line_splitter.split_documents(docs)
    print(f"拆分后文档数: {len(split_docs)}")

    # 4) 初始化 Embedding (Qwen + MLP)
    embedding_model = QwenMathEmbedding(qwen_model_path, mlp_weight_path, device="cuda")

    # 5) 构建Chroma向量库
    from langchain.vectorstores import Chroma
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    print(f"已创建并保存Chroma数据库到: {persist_dir}")

    # 6) 简单测试
    query = "函数的概念"
    results = vectorstore.similarity_search(query, k=2)
    print(f"\n查询: {query}")
    for i, doc in enumerate(results):
        print(f"[{i+1}] 相似片段:\n{doc.page_content}")

if __name__ == "__main__":
    main()
