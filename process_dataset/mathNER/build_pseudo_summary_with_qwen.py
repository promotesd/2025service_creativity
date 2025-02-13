import os
import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def auto_explain_bracketed_term(qwen_model, qwen_tokenizer, device, term: str, word_limit=50) -> str:
    """
    用 Qwen 对括号中的词语做解释/定义, 不超过 word_limit 字.
    term 不含【】，例如 "函数"、"集合" 等
    """
    messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {
            "role": "user",
            "content": (
                f"请用中文简要解释【{term}】在高中数学中的定义或含义，字数不超过{word_limit}字。"
            ),
        },
    ]
    text_prompt = qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = qwen_tokenizer([text_prompt], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = qwen_model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=False  # 可改True+temperature
        )
    input_len = model_inputs["input_ids"].shape[1]
    output_ids = generated_ids[0][input_len:]
    explanation = qwen_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return explanation

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = '/root/autodl-tmp/model/Qwen2-math7B'
    print("Loading Qwen model...")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    qwen_tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 输入文本
    input_file = "/root/autodl-tmp/code/2025service_creativity/process_dataset/mathNER/test.txt"
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return

    # 输出 JSON
    output_json = "/root/autodl-tmp/code/2025service_creativity/process_dataset/mathNER/test.json"

    # 读取文本
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 用正则匹配所有括号【xxx】
    pattern = re.compile(r"【(.*?)】")
    bracketed_terms = pattern.findall(content)
    print(f"在文本中找到 {len(bracketed_terms)} 处【...】内容。")

    # (可选) 是否去重
    bracketed_terms = list(set(bracketed_terms))
    print(f"去重后剩余 {len(bracketed_terms)} 个词。")

    # 对每个括号中的词调用Qwen生成解释
    results = []
    for i, term in enumerate(bracketed_terms):
        # 前面匹配到的 term 已不含【】，直接使用
        explanation = auto_explain_bracketed_term(qwen_model, qwen_tokenizer, device, term, word_limit=50)
        results.append({
            "src": f"【{term}】",
            "tgt": explanation
        })
        if (i+1) % 5 == 0:
            print(f"已处理 {i+1}/{len(bracketed_terms)} 个词: {term} => {explanation[:20]}...")

    # 写出 JSON
    with open(output_json, "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n已生成括号词语解释文件: {output_json}, 共{len(results)}条记录.")

if __name__ == "__main__":
    main()
