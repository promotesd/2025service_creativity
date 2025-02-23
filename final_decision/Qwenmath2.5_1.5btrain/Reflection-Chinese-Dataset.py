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

# ------------------------------------------------
# A) 回调1：每个整Epoch保存 & 验证
# ------------------------------------------------
class SaveEvery12EpochsCallback(TrainerCallback):
    """
    在每个整 epoch 结束时( epoch>=1 ):
      - control.should_save = True
      - control.should_evaluate = True
    => Trainer 会执行 _save_checkpoint() 并 evaluate()
    
    如想“每12个epoch保存”，可在if中改 (epoch_int%12)==0
    """
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        epoch_int = int(state.epoch)
        if epoch_int>0:
            control.should_save = True
            control.should_evaluate = True
        else:
            control.should_save=False
        return control


# ------------------------------------------------
# B) 回调2：验证后打印2条生成示例
# ------------------------------------------------
class EvalPrintSamplesCallback(TrainerCallback):
    """
    每次验证后:
      - 随机抽2条验证集
      - 模型generate => 打印
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
        if (model is None) or (tokenizer is None):
            return
        trainer = kwargs.get("trainer", None)
        if trainer is None:
            return

        val_ds = trainer.eval_dataset
        if not val_ds or len(val_ds)==0:
            return

        sample_idxes = random.sample(range(len(val_ds)), min(2, len(val_ds)))
        print("\n[EvalPrintSamplesCallback] sample 2 from validation set:\n")
        for idx in sample_idxes:
            ex=val_ds[idx]
            text_inp= ex["qwen_formatted_text"]
            short_inp=text_inp[:300]
            if len(text_inp)>300:
                short_inp+="..."
            print(f" [val idx={idx}] partial input: {short_inp}")

            inputs = tokenizer(
                text_inp,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            for k in inputs:
                inputs[k]=inputs[k].to(model.device)

            gen_out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                top_p=0.9,
                temperature=0.8
            )
            gen_text= tokenizer.decode(gen_out[0], skip_special_tokens=True)
            print(f" => generation:\n{gen_text}\n")


# ------------------------------------------------
# C) 数据集加载 => “stvlynn/Reflection-Chinese-Dataset”
# ------------------------------------------------
def load_reflection_chinese_dataset(dataset_name="stvlynn/Reflection-Chinese-Dataset") -> DatasetDict:
    """
    通常仅有 train(9171 行)
    """
    ds = load_dataset(dataset_name)
    if "train" not in ds:
        raise ValueError(f"Dataset must have 'train'. Found: {list(ds.keys())}.")
    return ds


def convert_to_chatml(example: Dict[str,Any]) -> Dict[str,str]:
    """
    prompt => user
    response => assistant
    => <|im_start|>user ...<|im_end|>\n<|im_start|>assistant ...<|im_end|>
    
    如果 response 中含 <thinking>, <reflection> 等, 保留
    """
    user_prompt= example.get("prompt","").strip()
    assistant_resp= example.get("response","").strip()

    user_part= f"<|im_start|>user {user_prompt}<|im_end|>\n"
    assistant_part= f"<|im_start|>assistant {assistant_resp}<|im_end|>"

    return {"qwen_formatted_text": user_part+assistant_part}


def split_train_val(ds: DatasetDict, val_ratio=0.1, seed=42)-> DatasetDict:
    """
    对 ds['train'] 做9:1拆分 => (train, validation)
    """
    train_all= ds["train"]
    splitted= train_all.train_test_split(test_size=val_ratio, seed=seed)
    return DatasetDict({
        "train": splitted["train"],
        "validation": splitted["test"]
    })


def save_dataset_as_json(dataset, file_path:str, text_col="qwen_formatted_text"):
    recs=[]
    for ex in dataset:
        recs.append({text_col: ex[text_col]})
    with open(file_path,"w",encoding="utf-8") as f:
        json.dump(recs,f,ensure_ascii=False,indent=2)


# ------------------------------------------------
# D) DataCollator + Metrics
# ------------------------------------------------
@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer
    max_length:int=1024

    def __call__(self, features:List[Dict[str,Any]])->Dict[str,torch.Tensor]:
        texts= [f["qwen_formatted_text"] for f in features]
        enc= self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        enc["labels"]= enc["input_ids"].clone()
        return enc

def compute_metrics_for_lm(eval_pred):
    """
    predictions => logits [batch, seq_len, vocab_size]
    label_ids => [batch, seq_len]
    统计token-level accuracy
    """
    logits, labels= eval_pred.predictions, eval_pred.label_ids
    preds= logits.argmax(dim=-1)
    valid_mask= (labels!=-100)
    correct= ((preds==labels)&valid_mask).sum().item()
    total= valid_mask.sum().item()
    accuracy= correct/max(total,1)
    return {"accuracy":accuracy}


# ------------------------------------------------
# E) LoRA 微调函数
# ------------------------------------------------
def train_qwen_lora(
    model_path:str,
    output_dir:str,
    train_dataset,
    val_dataset,
    num_train_epochs:int=32,
    eval_each_epoch=True,
    callbacks:List[TrainerCallback]=None,
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    max_seq_length=1024,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.05
):
    print("[INFO] Loading base Qwen from", model_path)
    tokenizer= AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # (可选)把 <thinking>, <reflection>, <output> 当作 special tokens
    new_tokens= ["<thinking>","</thinking>","<reflection>","</reflection>","<output>","</output>"]
    special_tokens_dict= {"additional_special_tokens": new_tokens}
    added_num= tokenizer.add_special_tokens(special_tokens_dict)
    print(f"[INFO] Added {added_num} new special tokens:", new_tokens)

    base_model= AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    # 若添加了新token => resize
    if added_num>0:
        base_model.resize_token_embeddings(len(tokenizer))
        print(f"[INFO] Resized embeddings => {len(tokenizer)} tokens")

    # 1) LoRA注入
    from peft import LoraConfig, get_peft_model, TaskType
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"]
    lora_config= LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM
    )
    print("[INFO] LoRA config:", lora_config)
    lora_model= get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()

    data_collator= DataCollatorForCausalLM(tokenizer, max_seq_length)

    from transformers import TrainingArguments, Trainer
    training_args= TrainingArguments(
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
        save_strategy="no", # rely on callback
        save_total_limit=3,
        remove_unused_columns=False
    )

    trainer= Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks or [],
        compute_metrics=compute_metrics_for_lm
    )

    print("[INFO] Start LoRA training reflection dataset ...")
    trainer.train()
    print("[INFO] Training done.")

    print("[INFO] Saving final LoRA model & tokenizer ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[INFO] Done. LoRA model saved to:", output_dir)


# ------------------------------------------------
# 5) 主函数
# ------------------------------------------------
def main():
    base_path= "/root/autodl-tmp/code/2025service_creativity/train_weight/reflection_chinese_lora"
    ts= datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir= os.path.join(base_path,f"train_lora_{ts}")
    os.makedirs(output_dir,exist_ok=True)

    # 1) 加载 reflection-chinese
    ds= load_reflection_chinese_dataset("stvlynn/Reflection-Chinese-Dataset")
    # => {train:9171,...}
    
    # 2) 转 chatml
    ds= ds.map(
        convert_to_chatml,
        desc="Convert prompt,response => Qwen ChatML"
    )

    # 3) split => train,val(9:1)
    from datasets import DatasetDict
    ds_dict= DatasetDict({
        "train": ds["train"]
    })
    ds_dict= split_train_val(ds_dict,val_ratio=0.1,seed=42)
    # => {train(8253), validation(918)}

    # 4) 导出
    if "train" in ds_dict:
        save_dataset_as_json(ds_dict["train"], os.path.join(output_dir,"train.json"))
    if "validation" in ds_dict:
        save_dataset_as_json(ds_dict["validation"], os.path.join(output_dir,"val.json"))

    # 5) 回调 => SaveEvery12EpochsCallback + EvalPrint
    class EvalPrintSamplesCallback(TrainerCallback):
        def on_evaluate(self,args,state,control,model=None,tokenizer=None,**kwargs):
            if model is None or tokenizer is None:
                return
            trainer=kwargs.get("trainer",None)
            if trainer is None:
                return
            val_ds= trainer.eval_dataset
            if len(val_ds)==0:
                return
            sample_idxes= random.sample(range(len(val_ds)), min(2,len(val_ds)))
            print("\n[EvalPrintSamplesCallback] sample 2 from val:\n")
            for idx in sample_idxes:
                ex= val_ds[idx]
                text_inp= ex["qwen_formatted_text"]
                short_inp=text_inp[:300]
                if len(text_inp)>300:
                    short_inp+="..."
                print(f" [val idx={idx}] partial input: {short_inp}")
                inputs= tokenizer(text_inp,return_tensors="pt",truncation=True,max_length=1024)
                for k in inputs:
                    inputs[k]=inputs[k].to(model.device)
                gen_out=model.generate(**inputs,max_new_tokens=64,do_sample=True,top_p=0.9,temperature=0.8)
                gen_text= tokenizer.decode(gen_out[0],skip_special_tokens=True)
                print(f" => generation:\n{gen_text}\n")

    save_cb= SaveEvery12EpochsCallback()
    eval_cb= EvalPrintSamplesCallback()
    callbacks=[save_cb, eval_cb]

    # 6) LoRA微调
    model_path= "/root/autodl-tmp/model/Qwenmath_2.5_1.5b" # 替换为你的Qwen路径
    train_ds= ds_dict["train"]
    val_ds= ds_dict["validation"]

    train_qwen_lora(
        model_path=model_path,
        output_dir=output_dir,
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_train_epochs=32,   # demo
        eval_each_epoch=True,
        callbacks=callbacks,
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        max_seq_length=1024,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05
    )


if __name__=="__main__":
    main()
