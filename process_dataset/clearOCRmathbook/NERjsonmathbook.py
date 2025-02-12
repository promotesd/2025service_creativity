import os
import json
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========== 0. 基本配置 ==========
DATA_DIR = r"/root/autodl-tmp/code/2025service_creativity/process_dataset/OCRmathbook"
OUTPUT_JSON = r"/root/autodl-tmp/code/2025service_creativity/process_dataset/clearOCRmathbook/NERjsonmathbook.json"

MODEL_PATH = r"/root/autodl-tmp/model/Qwen2-math7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 设置并发数，如果出问题可设置为1做串行
MAX_WORKERS = 1

# ========== 1. 加载模型与分词器 ==========
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 如果没有 apply_chat_template，就自定义一个简易函数
if not hasattr(tokenizer, "apply_chat_template"):
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
        text_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text_parts.append(f"<{role}>: {content}")
        if add_generation_prompt:
            text_parts.append("<assistant>:")
        return "\n".join(text_parts)

    tokenizer.apply_chat_template = apply_chat_template


# ========== 2. 构造 Prompt ==========

def build_prompt(text_chunk: str, source_name: str) -> str:
    """
    生成对话 Prompt，指导模型做命名实体识别并输出 JSON 数组。
    """
    system_message = (
        "你是一位擅长数学文本解析的AI助手。下面的文本来自一篇数学教材，"
        "其中可能包含定理、定义、数学术语、重要公式等。"
        "请你抽取这些实体，并输出 JSON 数组，形如：\n"
        "["
        "  {\"type\": \"定理\", \"content\": \"...\", \"formula\": \"...\", \"source\": \"...\"},\n"
        "  {\"type\": \"定义\", \"content\": \"...\", \"source\": \"...\"},\n"
        "  ...\n"
        "]\n"
        "如果没有可识别实体，就返回空数组 []。"
        "不要输出其他内容，严格返回 JSON。"
    )

    user_message = (
        f"文本来源: {source_name}\n"
        f"文本内容:\n\n"
        f"{text_chunk}\n\n"
        "请识别其中的定理、定义、数学术语、公式，并以 JSON 数组形式返回。"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt_text

# ========== 3. 调用模型并解析 JSON ==========

def run_ner_on_chunk(text_chunk: str, source_name: str) -> list:
    """
    对一个文本块调用模型做命名实体识别，若解析失败或结果无效，则返回 []。
    """
    prompt_text = build_prompt(text_chunk, source_name)

    inputs = tokenizer([prompt_text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,  # 可视需求调
            do_sample=False
        )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0][input_len:]
    raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # 解析 JSON
    try:
        parsed = json.loads(raw_output)
        # 检查是否是 list
        if not isinstance(parsed, list):
            return []  # 不是列表则跳过
        # 给所有实体补上 source
        for entity in parsed:
            if isinstance(entity, dict):
                entity.setdefault("source", source_name)
        return parsed
    except:
        # 如果解析失败，就返回空列表（不保存）
        return []

# ========== 4. 追加到 JSON 文件 ==========

def append_to_json_file(data: list, filename: str):
    """
    data: list of entity objects (不为空才写)
    以追加方式写入 JSON 文件中的大数组。
    """
    if not data:
        return  # 如果空列表，直接跳过

    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with open(filename, "r", encoding="utf-8") as f:
            try:
                old_list = json.load(f)
                if not isinstance(old_list, list):
                    old_list = []
            except:
                old_list = []
        old_list.extend(data)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(old_list, f, ensure_ascii=False, indent=2)

# ========== 5. 遍历文件，分块处理并写入 ==========

def main():
    # 如果存在旧的 OUTPUT_JSON，可以删除或保留
    if os.path.exists(OUTPUT_JSON):
        os.remove(OUTPUT_JSON)

    # 找到所有 .txt 文件
    txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

    for i, txt_file in enumerate(txt_files):
        file_path = os.path.join(DATA_DIR, txt_file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 分块避免 prompt 太长
        chunk_size = 1024
        start_idx = 0
        while start_idx < len(content):
            end_idx = min(start_idx + chunk_size, len(content))
            text_chunk = content[start_idx:end_idx]
            start_idx = end_idx

            # 调用 NER
            entities = run_ner_on_chunk(text_chunk, txt_file)
            # 追加写入
            append_to_json_file(entities, OUTPUT_JSON)

        print(f"[INFO] 已处理文件: {txt_file} (进度 {i+1}/{len(txt_files)})")

    print(f"处理完成！输出文件: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
