#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_ppo_rlhf.py

示例: 在已经 SFT 微调好的 Qwen2-Math7B + LoRA 模型基础上,
进行一个简单的 PPO (Proximal Policy Optimization) 过程, 
用RewardModel打分 => 强化学习更新策略(LoRA).

注意:
 1) 需要 pip install trl>=0.4.7
 2) 需要一个真实 RewardModel 或人类打分. 这里只是 dummy 演示
 3) Qwen2-Math 可能很大, 需足够显存 + partial freeze
"""

import os
import torch
import random
import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig
from trl import PPOConfig, PPOTrainer
from trl.core import LengthSampler

###################################################
# 1) Dummy RewardModel
###################################################
class DummyRewardModel(torch.nn.Module):
    """
    演示: 真实情况下, 你会加载/训练好的RM, 
    例如 input: (context, generation) => scalar reward
    这里仅简单: 如果回答含"函数" => reward=+1, else=0
    """
    def __init__(self):
        super().__init__()
        # 仅做占位
    def forward(self, text_batch:List[str]) -> torch.Tensor:
        # text_batch: list of generated answers
        # Return a tensor of shape [batch_size], each is reward
        rews = []
        for txt in text_batch:
            if "函数" in txt:
                rews.append(1.0)
            else:
                rews.append(0.0)
        return torch.tensor(rews, dtype=torch.float)

###################################################
# 2) Self-consistency: generate multiple => pick best
###################################################
def generate_self_consistency(policy_model, tokenizer, query_text, num_samples=3, max_new_tokens=128):
    """
    一次对同一Query生成多条回答 => 由 RewardModel打分 => 选最高score
    """
    input_ids = tokenizer(query_text, return_tensors="pt").to(policy_model.device)
    candidates = []
    for _ in range(num_samples):
        out = policy_model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        prompt_len = input_ids.input_ids.shape[1]
        gen_ids = out[0][prompt_len:]
        ans_str= tokenizer.decode(gen_ids, skip_special_tokens=True)
        candidates.append(ans_str)
    return candidates

###################################################
# 3) PPO main
###################################################
def main():
    # 0) Load SFT policy (Qwen + LoRA)
    base_model_path= "/root/autodl-tmp/model/Qwen2-math7B"
    lora_weight_path= "/root/autodl-tmp/code/2025service_creativity/Qwen-math_train/lora_weight_final"  # 你SFT好的LoRA
    tokenizer= AutoTokenizer.from_pretrained(base_model_path)
    base_model= AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype="auto")
    # merge or keep LoRA
    policy_lora= PeftModel.from_pretrained(base_model, lora_weight_path)
    # merge
    policy_model= policy_lora.merge_and_unload()
    policy_model.eval()

    # 1) RewardModel
    reward_model= DummyRewardModel().cuda()

    # 2) PPO Config
    config= PPOConfig(
        batch_size=1,
        forward_batch_size=1,
        ppo_epochs=1,
        learning_rate=1e-6,
        log_with=None,
    )
    # 3) PPOTrainer
    ppo_trainer= PPOTrainer(
        config=config,
        model=policy_model,
        ref_model=None,   # 也可指定policy copy
        tokenizer=tokenizer,
    )

    # 4) RL training data
    #   例: offline 2条
    rl_data= [
      {"query":"请解释【函数】概念?", "ref_answer":"函数是..."},
      {"query":"什么是【导数】?",     "ref_answer":"导数表示函数的变化率..."}
    ]
    # 真实: 可能在线+human feedback

    # 5) RL Loop
    for step_i, sample in enumerate(rl_data):
        query_str= sample["query"]
        # multi generation => pick best
        cands= generate_self_consistency(ppo_trainer.model, tokenizer, query_str, num_samples=3)
        # 6) reward
        with torch.no_grad():
            rews= reward_model(cands)  # shape [3]
        best_idx= int(torch.argmax(rews))
        best_ans= cands[best_idx]
        best_reward= rews[best_idx].item()
        
        # PPO step => single (query->best_ans)
        queries= [query_str]
        responses= [best_ans]
        reward_t= torch.tensor([best_reward], dtype=torch.float, device=ppo_trainer.model.device)
        ppo_trainer.step(queries, responses, reward_t)
        
        print(f"[step {step_i}] best_ans: {best_ans[:40]}..., reward={best_reward}")

    # 7) save final
    final_out= "/root/autodl-tmp/code/2025service_creativity/Qwen-math_train/ppo_final"
    os.makedirs(final_out, exist_ok=True)
    ppo_trainer.model.save_pretrained(final_out)
    tokenizer.save_pretrained(final_out)
    print("[PPO done] final policy =>", final_out)

if __name__=="__main__":
    try:
        from trl import PPOTrainer, PPOConfig
        main()
    except ImportError:
        print("Please install `trl>=0.4.7`: pip install trl")
