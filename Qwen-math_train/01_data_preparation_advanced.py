#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_data_preparation_advanced.py

目的：
- 读取多轮对话数据 (其中assistant消息包含 graph_of_thought/reflexion/final_answer)
- 保留完整对话结构(多轮)以及 GoT/Reflexion/FinalAnswer
- 以 conversation-level 随机打乱并切分 train/val/test
- 最终输出:
    train.json / val.json / test.json
  其中每个元素是一条 conversation, messages中包含 role=user/assistant 等

这样你可在下游做多轮对话SFT、GoT图结构推理等
"""

import os
import json
import random

def main():
    src_file = "/root/autodl-tmp/code/2025service_creativity/process_dataset/huggingfaceset/ConvGoTAzure99_blossom-math-v1.json"
    out_dir = "/root/autodl-tmp/code/2025service_creativity/Qwen-math_train/ConvGoTAzure99_blossom-math-v1"
    os.makedirs(out_dir, exist_ok=True)

    with open(src_file, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    print(f"Loaded {len(conversations)} conversations from {src_file}")

    # 这里按 conversation 级别随机打乱
    random.shuffle(conversations)
    n = len(conversations)
    train_ratio, val_ratio = 0.8, 0.1
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train_data = conversations[:train_end]
    val_data   = conversations[train_end:val_end]
    test_data  = conversations[val_end:]

    # 将 train_data, val_data, test_data 写出
    def save_json(data_list, filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)

    train_out = os.path.join(out_dir, "train.json")
    val_out   = os.path.join(out_dir, "val.json")
    test_out  = os.path.join(out_dir, "test.json")

    save_json(train_data, train_out)
    save_json(val_data,   val_out)
    save_json(test_data,  test_out)

    print(f"Done! splitted into train/val/test in {out_dir}")
    print(f"train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

if __name__ == "__main__":
    main()
