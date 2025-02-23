import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ========== 0. 基本配置 ==========

MODEL_PATH = r"/root/autodl-tmp/model/Qwen2-math7B"  # 替换为你的模型目录或模型名称
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 输出文件，用于存储多轮对话JSON
OUTPUT_FILE = r"/root/autodl-tmp/code/2025service_creativity/process_dataset/huggingfaceset/ConvGoTAzure99_blossom-math-v1.json"

# ========== 1. 加载你的数据集 ==========

# 假设 dataset 包含 train / test / validation 等 split
# 在此处只演示对 "train" split 进行处理：
dataset_all = load_dataset(r"/root/.cache/huggingface/datasets/Azure99___blossom-math-v1")
print(dataset_all)  # 查看数据集信息

# 这里假设我们只处理 train split：
ds = dataset_all["train"]

# 如果你的数据集中字段并非 "input", "output", "solution"，请自行更改下面的逻辑
# 假设 ds 中每条记录如下：
# {
#   "id": ...,         # 题目编号
#   "input": ...,      # 用户题目
#   "output": ...,     # 参考输出
#   "solution": ...    # 参考解答
# }

# ========== 2. 准备示例对话 JSON，用作“结构参考” ==========

example_got_json = r"""
{
  "conversation_id": "1",
  "messages": [
    {
      "role": "user",
      "content": "勾股定理的详细推导是什么？能不能举例说明？"
    },
    {
      "role": "assistant",
      "content": "勾股定理主要阐述了直角三角形斜边与两直角边的平方和关系...",
      "graph_of_thought": {
        "nodes": [
          { "id": "n1", "content": "回忆定义：a^2 + b^2 = c^2, c是斜边" },
          { "id": "n2", "content": "推导一：利用面积拆分方法" },
          { "id": "n3", "content": "推导二：借助相似三角形" },
          { "id": "n4", "content": "举例：3,4,5" }
        ],
        "edges": [
          { "from": "n1", "to": "n2", "relation": "alternative_proof_1" },
          { "from": "n1", "to": "n3", "relation": "alternative_proof_2" },
          { "from": "n1", "to": "n4", "relation": "example" }
        ],
        "reflexion": [
          { 
            "node_id": "n3", 
            "self_check": "相似三角形推导是否清晰？", 
            "revised_content": "可以再补充角度关系..."
          }
        ]
      },
      "final_answer": "勾股定理的基本推导方法包括面积拼接法和相似三角形法，典型例子有3,4,5三元组..."
    }
  ],
  "summaries": [],
  "global_summary": "本段对话针对勾股定理的推导过程与整数例子进行较详细讨论。"
}
""".strip()


# ========== 3. 初始化模型和分词器 ==========

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"  # 如果有多GPU，可自动分配
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# ========== 4. 构造中文 Prompt：将 input/output/solution 合并进来 ==========

def build_prompt(record_id: str, user_input: str, std_output: str, std_solution: str) -> str:
    """
    使用中文提示，让模型输出与 example_got_json 类似的多轮对话格式。
    
    参数:
    - record_id: 题目编号或唯一ID
    - user_input: 用户提问/题目
    - std_output: 参考输出
    - std_solution: 参考解答
    """
    system_content = (
        "你是一位乐于助人的AI助手。下面有一个对话JSON的示例，它包含了以下内容：\n\n"
        f"{example_got_json}\n\n"
        "请仔细分析上面的JSON结构。现在有一个新题目，需要你产出一个类似的多轮对话JSON，要求：\n"
        "1. 包含一个独特的 \"conversation_id\"（例如可以使用题目的ID）。\n"
        "2. 在 \"messages\" 数组中，至少包含一条用户提问 (role=user) 和一条AI回答 (role=assistant)。\n"
        "3. 在 assistant 的回答中，一定要包含 \"graph_of_thought\" 字段（其中包含若干 \"nodes\", \"edges\", \"reflexion\" 等）\n"
        "4. assistant 的回答还需包含一个 \"final_answer\" 字段，用于给出最终简明结论。\n"
        "5. 你可以根据参考输出和解答内容进行总结、提炼、补充，但生成的 JSON 要**结构完整**、**逻辑严密**、**推理科学**，不要输出多余说明。\n"
        "6. 不要在 JSON 之外输出额外内容。只需返回纯 JSON。\n"
    )

    # 将题目信息、参考输出和参考解答，放在用户提示中
    user_content = (
        f"新的题目ID: {record_id}\n"
        f"题目内容 (user_input): {user_input}\n"
        f"参考输出 (std_output): {std_output}\n"
        f"参考解答 (std_solution): {std_solution}\n\n"
        "请你基于以上信息，输出一个完整的多轮对话JSON。"
    )

    # 拼接对话Prompt
    prompt_text = (
        f"<system>: {system_content}\n"
        f"<user>: {user_content}\n"
        "<assistant>:"  # 给模型一个标记，以便开始回答
    )
    return prompt_text


# ========== 5. 调用模型并尝试解析成 JSON ==========

def generate_got_json(prompt_text: str) -> dict:
    """
    给定中文 Prompt，调用模型生成文本，并尝试解析为 JSON。
    如果解析失败，则返回 { "raw_text": ... }。
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

    # 推理生成
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,  # 这里使用greedy，可改为True并设置温度等
            # top_k=1,
        )

    # 将输出转成字符串，并截掉Prompt的部分
    input_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0][input_len:]
    raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # 尝试解析JSON
    try:
        parsed_json = json.loads(raw_output)
        return parsed_json
    except Exception:
        return {"raw_text": raw_output}


# ========== 6. 工具函数：追加数据到 JSON 文件 ==========

def append_data_to_json_file(data: dict, filename: str):
    """
    读取原文件(若有)，将新数据追加到数组中，并写回。
    文件格式: [
      { conversation_json1... },
      { conversation_json2... },
      ...
    ]
    """
    if not os.path.exists(filename):
        # 如果文件不存在，创建一个新数组并写入
        with open(filename, "w", encoding="utf-8") as f:
            json.dump([data], f, ensure_ascii=False, indent=2)
    else:
        # 文件存在，则读取后追加
        with open(filename, "r", encoding="utf-8") as f:
            try:
                old_data = json.load(f)
                if not isinstance(old_data, list):
                    old_data = []
            except:
                old_data = []

        old_data.append(data)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(old_data, f, ensure_ascii=False, indent=2)


# ========== 7. 遍历数据集并生成 GoT 对话 JSON，保存到文件 ==========

def main():
    # 这里可以先清空或初始化输出文件
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)  # 如果想保留历史记录，可以不删除

    # 遍历数据集中的每条样本
    for i, row in enumerate(ds):
        # 假设字段为["id", "input", "output", "solution"]
        record_id = str(row["id"])
        user_input = row["input"]
        std_output = row["output"]
        std_solution = row["answer"]

        # 构造中文 prompt
        prompt_text = build_prompt(record_id, user_input, std_output, std_solution)

        # 调用模型生成
        conversation_json = generate_got_json(prompt_text)

        # 追加到输出文件
        append_data_to_json_file(conversation_json, OUTPUT_FILE)

        # 仅演示前10条，可自行去掉限制
        if i < 10:
            print(f"[INFO] 已生成第 {i+1} 条记录，conversation_id={record_id}")
        if i == 9:
            print("（仅打印前10条处理情况，其余继续处理中...）")

    print(f"处理完成！所有 GoT 对话结果保存在 {OUTPUT_FILE}。")


if __name__ == "__main__":
    main()
