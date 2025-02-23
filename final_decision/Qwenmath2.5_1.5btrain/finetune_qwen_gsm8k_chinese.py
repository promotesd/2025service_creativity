#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import json
import random
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass

import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import LoraConfig, get_peft_model, TaskType

# ------------------------------------------------------
# A) Callback: 每个整 epoch 保存 + 验证
# ------------------------------------------------------
class SaveEveryEpochCallback(TrainerCallback):
    """
    在每个 epoch 结束时:
      - should_save = True
      - should_evaluate = True
    => Trainer会做 _save_checkpoint() 并 evaluate()
    """
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        epoch_int = int(state.epoch)
        if epoch_int > 0:
            control.should_save = True
            control.should_evaluate = True
        else:
            control.should_save = False
        return control


# ------------------------------------------------------
# B) Callback: 验证后打印2条生成示例
# ------------------------------------------------------
class EvalPrintSamplesCallback(TrainerCallback):
    """
    在 on_evaluate 时:
      - 随机抽2条验证集
      - 生成&打印
    """
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        tokenizer=None,
        **kwargs
    ):
        if model is None or tokenizer is None:
            return

        trainer = kwargs.get("trainer", None)
        if trainer is None:
            return

        val_ds = trainer.eval_dataset
        if not val_ds or len(val_ds)==0:
            return

        sample_idxes = random.sample(range(len(val_ds)), min(2, len(val_ds)))
        print("\n[EvalPrintSamplesCallback] sample 2 from val:\n")
        for idx in sample_idxes:
            ex=val_ds[idx]
            text_inp=ex["qwen_formatted_text"]
            short_inp=text_inp[:300]
            if len(text_inp)>300:
                short_inp+="..."
            print(f" [val idx={idx}] input partial: {short_inp}")

            inputs=tokenizer(
                text_inp, return_tensors="pt",
                truncation=True, max_length=1024
            )
            for k in inputs:
                inputs[k] = inputs[k].to(model.device)

            gen_out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                top_p=0.9,
                temperature=0.8
            )
            gen_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
            print(f" => generation:\n{gen_text}\n")


# ------------------------------------------------------
# C) 加载 + 抽样: meta-math/MetaMathQA_GSM8K_zh
# ------------------------------------------------------
def load_and_sample_metamath(
    dataset_name="meta-math/MetaMathQA_GSM8K_zh",
    sample_size=8000,
    seed=42
)-> DatasetDict:
    ds = load_dataset(dataset_name)
    if "train" not in ds:
        raise ValueError(f"Dataset must have 'train'. Found: {list(ds.keys())}.")

    train_all = ds["train"]
    total_count = train_all.num_rows
    if sample_size> total_count:
        sample_size=total_count
    train_sample = train_all.shuffle(seed=seed).select(range(sample_size))

    new_dict = {"train":train_sample}
    if "test" in ds:
        new_dict["test"]=ds["test"]
    if "validation" in ds:
        new_dict["validation"]=ds["validation"]
    return DatasetDict(new_dict)


def make_multiturn_dialogue(example: Dict[str,Any]) -> Dict[str,str]:
    """
    构造2轮 ChatML:
    1) user => {query_zh}
    2) assistant => "[GoT提示]...\n[Chain-of-Thought]: response_zh"
    """
    query= example.get("query_zh","").strip()
    resp = example.get("response_zh","").strip()
    if not query and not resp:
        return {"qwen_formatted_text":""}

    user_txt = f"<|im_start|>user {query}<|im_end|>\n"
    assistant_txt = "[GoT提示]: 让我们先理解题意并逐步推理.\n"
    assistant_txt+= f"[Chain-of-Thought]: {resp}\n"
    assistant_part = f"<|im_start|>assistant {assistant_txt}<|im_end|>"

    return {"qwen_formatted_text": user_txt + assistant_part}


def split_train_val(dataset_dict:DatasetDict, val_ratio=0.1, seed=42)->DatasetDict:
    if "train" not in dataset_dict:
        raise ValueError("'train' not found in dataset.")
    train_ds=dataset_dict["train"]
    splitted = train_ds.train_test_split(test_size=val_ratio, seed=seed)
    new_dict = {
        "train": splitted["train"],
        "validation": splitted["test"]
    }
    if "test" in dataset_dict:
        new_dict["test"] = dataset_dict["test"]
    return DatasetDict(new_dict)


def save_dataset_as_json(dataset, file_path:str, text_col="qwen_formatted_text"):
    recs=[]
    for ex in dataset:
        recs.append({text_col: ex[text_col]})
    with open(file_path,"w",encoding="utf-8") as f:
        json.dump(recs,f,ensure_ascii=False,indent=2)


# ------------------------------------------------------
# D) Collator + Metrics
# ------------------------------------------------------
@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer
    max_length:int=1024
    def __call__(self,features:List[Dict[str,Any]])->Dict[str,torch.Tensor]:
        texts= [ ex["qwen_formatted_text"] for ex in features ]
        enc= self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        enc["labels"]=enc["input_ids"].clone()
        return enc

def compute_metrics_for_lm(eval_pred):
    """
    eval_pred.predictions => [batch, seq_len, vocab_size] logits
    eval_pred.label_ids   => [batch, seq_len]
    计算token-level accuracy
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = logits.argmax(dim=-1)  # [batch, seq_len]
    valid_mask=(labels!=-100)
    correct=((preds==labels)&valid_mask).sum().item()
    total=valid_mask.sum().item()
    accuracy = correct/max(total,1)
    return {"accuracy":accuracy}


# ------------------------------------------------------
# E) LoRA 微调函数
# ------------------------------------------------------
def train_qwen_lora(
    model_path:str,
    output_dir:str,
    train_dataset,
    val_dataset,
    num_train_epochs:int=12,
    eval_each_epoch:bool=True,
    callbacks:List[TrainerCallback]=None,
    learning_rate:float=1e-4,
    per_device_train_batch_size:int=4,
    per_device_eval_batch_size:int=4,
    max_seq_length:int=1024,
    lora_rank:int=8,
    lora_alpha:int=16,
    lora_dropout:float=0.05
):
    print("[INFO] Loading base Qwen + tokenizer from", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # 1) LoRA
    from peft import LoraConfig, get_peft_model, TaskType
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"]
    lora_config=LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM
    )
    print("[INFO] Lora Config:", lora_config)
    lora_model = get_peft_model(base_model,lora_config)
    lora_model.print_trainable_parameters()

    collator=DataCollatorForCausalLM(tokenizer, max_seq_length)

    from transformers import TrainingArguments, Trainer
    training_args=TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch" if eval_each_epoch else "no",
        logging_strategy="steps",
        logging_steps=50,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=1,
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
        save_strategy="no",
        save_total_limit=3,
        remove_unused_columns=False
    )

    trainer=Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics_for_lm,
        callbacks=callbacks
    )

    print("[INFO] Start LoRA training ...")
    trainer.train()
    print("[INFO] Training done")

    print("[INFO] Save final LoRA model & tokenizer ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[INFO] Done. LoRA model saved at:", output_dir)


# ------------------------------------------------------
# 6) main
# ------------------------------------------------------
def main():
    base_dir="/root/autodl-tmp/code/2025service_creativity/train_weight/metamathqa_gsm8k_zh"
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir=os.path.join(base_dir,f"train_lora_{ts}")
    os.makedirs(output_dir,exist_ok=True)

    # 1) load & sample
    ds=load_and_sample_metamath(
        dataset_name="meta-math/MetaMathQA_GSM8K_zh",
        sample_size=8000,
        seed=42
    )
    # => {train(8000), possibly test,validation...}

    # 2) map => build ChatML
    ds=ds.map(make_multiturn_dialogue, desc="Make multi-turn with GoT+CoT")

    # 3) split train => val(9:1)
    ds=split_train_val(ds,val_ratio=0.1, seed=42)

    # 4) dump to JSON
    if "train" in ds:
        save_dataset_as_json(ds["train"], os.path.join(output_dir,"train.json"))
    if "validation" in ds:
        save_dataset_as_json(ds["validation"], os.path.join(output_dir,"val.json"))
    if "test" in ds:
        save_dataset_as_json(ds["test"], os.path.join(output_dir,"test.json"))

    # 5) callbacks
    cb_save=SaveEveryEpochCallback()
    cb_eval=EvalPrintSamplesCallback()
    callbacks=[cb_save, cb_eval]

    # 6) LoRA finetune
    model_path="/root/autodl-tmp/model/Qwenmath_2.5_1.5b"  # 替换成你的Qwen路径
    train_ds=ds["train"]
    val_ds=ds["validation"]

    train_qwen_lora(
        model_path=model_path,
        output_dir=output_dir,
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_train_epochs=6,   # 仅示例
        eval_each_epoch=True,
        callbacks=callbacks,
        learning_rate=1e-4,  # LoRA通常可用稍大lr
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        max_seq_length=1024,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05
    )

if __name__=="__main__":
    main()
