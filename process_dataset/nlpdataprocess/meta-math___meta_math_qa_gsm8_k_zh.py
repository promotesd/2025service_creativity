import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ========== 0. 基本配置 ==========
MODEL_PATH = r"/root/autodl-tmp/model/Qwen2-math7B"  # 替换为你的模型目录或模型名称
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 输出文件，用于存储多轮对话JSON
OUTPUT_FILE = r"/root/autodl-tmp/code/2025service_creativity/process_dataset/huggingfaceset/GoTmeta-math_qa_gsm8_k_zh.json"

# ========== 1. 加载你的数据集 ==========

# 如果你要用 "meta-math___meta_math_qa_gsm8_k_zh" 数据集，就加载这个：
dataset_all = load_dataset(r"/root/.cache/huggingface/datasets/meta-math___meta_math_qa_gsm8_k_zh")

# 或者用 "Azure99___blossom-math-v1"，如你的原代码所示，可以自行切换
# dataset_all = load_dataset(r"/root/.cache/huggingface/datasets/Azure99___blossom-math-v1")

print(dataset_all)  # 查看数据集信息

# 这里假设我们只处理 train split
ds = dataset_all["train"]

# 你的数据集字段大约是 ['query', 'response', 'type', 'query_zh', 'response_zh']
# 其中 `query_zh` 是中文题目, `response_zh` 是中文解答

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
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# ========== 4. 构造中文 Prompt ==========
def build_prompt(record_id: str, user_input: str, std_output: str) -> str:
    """
    使用中文提示，让模型输出与 example_got_json 类似的多轮对话格式。
    
    参数:
    - record_id: 题目编号或唯一ID
    - user_input: 用户提问/题目 (这里对应 query_zh)
    - std_output: 参考输出 (你可以和 std_solution 一样，也可以做区分)
    - std_solution: 参考解答
    """
    system_content = (
        "你是一位乐于助人的AI助手。下面有一个对话JSON的示例，它包含了以下内容：\n\n"
        f"{example_got_json}\n\n"
        "请仔细分析上面的JSON结构。现在有一个新题目，需要你产出一个类似的多轮对话JSON，要求：\n"
        "1. 包含一个独特的 \"conversation_id\"。\n"
        "2. 在 \"messages\" 数组中，至少包含一条用户提问 (role=user) 和一条AI回答 (role=assistant)。\n"
        "3. 在 assistant 的回答中，一定要包含 \"graph_of_thought\" 字段（其中包含若干 \"nodes\", \"edges\", \"reflexion\" 等）\n"
        "4. assistant 的回答还需包含一个 \"final_answer\" 字段，用于给出最终简明结论。\n"
        "5. 你可以根据参考输出和解答内容进行总结、提炼、补充，但生成的 JSON 要**结构完整**、**逻辑严密**、**推理科学**，不要输出多余说明。\n"
        "6. 不要在 JSON 之外输出额外内容。只需返回纯 JSON。\n"
    )

    user_content = (
        f"题目内容 (user_input): {user_input}\n"
        f"参考解答 (std_output): {std_output}\n"
        "请你基于以上信息，输出一个完整的多轮对话JSON。"
    )

    prompt_text = (
        f"<system>: {system_content}\n"
        f"<user>: {user_content}\n"
        "<assistant>:"
    )
    return prompt_text


# ========== 5. 调用模型并尝试解析成 JSON ==========
def generate_got_json(prompt_text: str) -> dict:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,  # 可改为True并设置 temperature、top_k 等
        )

    # 将输出转成字符串，并截掉 Prompt
    input_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0][input_len:]
    raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # 尝试解析为 JSON
    try:
        parsed_json = json.loads(raw_output)
        return parsed_json
    except Exception:
        # 如果无法解析，则返回原始文本
        return {"raw_text": raw_output}


# ========== 6. 追加数据到 JSON 文件 ==========
def append_data_to_json_file(data: dict, filename: str):
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump([data], f, ensure_ascii=False, indent=2)
    else:
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


# ========== 7. 遍历数据集并生成 JSON ==========

def main():
    # 如果之前存在OUTPUT_FILE，可以视需求是否删除
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    # 遍历数据集中的每条样本
    for i, row in enumerate(ds):
        # 你的数据集里没有 "id" 字段的话，可以用 i 来生成 conversation_id
        record_id = f"mmath-{i+1}"

        # 重点：将 query_zh / response_zh 映射到我们的 prompt 参数里
        user_input = row["query_zh"]        # 中文题目
        std_output = row["response_zh"]     # 中文参考输出
        

        # 构造 Prompt
        prompt_text = build_prompt(record_id, user_input, std_output)

        # 调用模型生成
        conversation_json = generate_got_json(prompt_text)

        # 追加到输出文件
        append_data_to_json_file(conversation_json, OUTPUT_FILE)

        # 仅示例打印前10条
        if i < 10:
            print(f"[INFO] 第 {i+1} 条已处理，conversation_id={record_id}")
        if i == 9:
            print("（仅打印前10条处理情况，其余继续处理中...）")

    print(f"处理完成！所有 GoT 对话结果保存在 {OUTPUT_FILE}。")


if __name__ == "__main__":
    main()
