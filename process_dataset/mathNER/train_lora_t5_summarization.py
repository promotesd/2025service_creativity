import os
import torch
import json
import numpy as np
from typing import Dict, Any
from datasets import Dataset
from evaluate import load  # 从 evaluate 库加载 ROUGE
from transformers import (
    T5ForConditionalGeneration,
    BertTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# ========== 数据加载函数 ==========
def load_my_json_data(file_path):
    """
    读取 JSON 文件: [{"src": "原文...", "tgt": "目标文本..."}...]
    转换成 Hugging Face Dataset 格式
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

# ========== 数据collate函数 ==========
def data_collator(features, tokenizer):
    """
    对 (src, tgt) 做分词并构造模型所需的 input_ids, labels
    不需要 token_type_ids, 并使用 text_target 处理目标
    """
    src_texts = [f["src"] for f in features]
    tgt_texts = [f["tgt"] for f in features]

    # 对输入分词
    model_inputs = tokenizer(
        src_texts,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt"
    )
    # 对目标分词
    labels = tokenizer(
        text_target=tgt_texts,
        truncation=True,
        max_length=64,
        padding=True,
        return_tensors="pt"
    )

    # T5 不使用 token_type_ids
    if "token_type_ids" in model_inputs:
        model_inputs.pop("token_type_ids")
    if "token_type_ids" in labels:
        labels.pop("token_type_ids")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ========== 自定义评估指标(ROUGE) ==========
rouge_metric = load("rouge")

def compute_metrics(eval_preds):
    """
    评估ROUGE
    eval_preds: (predictions, label_ids)
    predictions 通常是 logits 或 token IDs
    """
    predictions, label_ids = eval_preds

    # 如果 predictions 是 tuple，就取第一个元素 (logits)
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # shape: [batch, seq_len, vocab_size]

    # 若是三维 [batch, seq_len, vocab_size]，需要 argmax
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)

    # 将 -100 替换成 pad_token_id 便于 decode
    label_ids = [[(token if token != -100 else 0) for token in seq] for seq in label_ids]

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"]
    }

# ========== 主函数 ==========

def main():
    # 1) 数据文件路径
    train_file = "/root/autodl-tmp/code/2025service_creativity/process_dataset/mathNER/train.json"
    val_file   = "/root/autodl-tmp/code/2025service_creativity/process_dataset/mathNER/val.json"
    test_file  = "/root/autodl-tmp/code/2025service_creativity/process_dataset/mathNER/test.json"

    # 2) 加载数据集
    train_ds = load_my_json_data(train_file)
    val_ds   = load_my_json_data(val_file)
    test_ds  = load_my_json_data(test_file)

    print("训练集大小:", len(train_ds))
    print("验证集大小:", len(val_ds))
    print("测试集大小:", len(test_ds))

    # 3) 加载本地 T5 模型 + T5Tokenizer
    base_model_name = "/root/autodl-tmp/model/t5-base-chinese-cluecorpussmall"
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(base_model_name)
    model = T5ForConditionalGeneration.from_pretrained(base_model_name)

    # 4) 配置 LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM"  
    )
    model = get_peft_model(model, peft_config)

    # 5) 训练配置
    output_dir = "/root/autodl-tmp/code/2025service_creativity/process_dataset/mathNER/t5_lora_out"
    lora_weight_dir = "/root/autodl-tmp/code/2025service_creativity/process_dataset/mathNER/t5_lora_weight"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(lora_weight_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4000,         # 训练轮数
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="steps",
        eval_steps=400,
        save_steps=400,
        logging_steps=400,
        learning_rate=1e-4,
        fp16=True,                   # 如果GPU支持
        overwrite_output_dir=True,
        remove_unused_columns=False
    )

    def collate_fn(batch):
        return data_collator(batch, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )

    # 6) 开始训练
    trainer.train()

    # 7) 保存 LoRA 权重
    model.save_pretrained(lora_weight_dir)
    print(f"LoRA 微调完成，权重保存在 {lora_weight_dir}")

    # 8) 在验证集上看最终评估分
    print("===== Final Evaluate on Val set =====")
    val_result = trainer.evaluate(eval_dataset=val_ds)
    print("Val result:", val_result)

    # 9) 在测试集上做最终评估
    print("===== Evaluate on Test set =====")
    test_result = trainer.evaluate(eval_dataset=test_ds)
    print("Test result:", test_result)

    # 10) 保存评估结果
    result_file = os.path.join(output_dir, "evaluation_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "val_result": {k: float(v) for k, v in val_result.items() if isinstance(v, (int, float))},
            "test_result": {k: float(v) for k, v in test_result.items() if isinstance(v, (int, float))}
        }, f, ensure_ascii=False, indent=2)

    print(f"评估结果已保存到 {result_file}")

if __name__ == "__main__":
    main()
