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
# 1) Callback: 每个 epoch 都保存 & 验证
# ------------------------------------------------------
class SaveEveryEpochCallback(TrainerCallback):
    """
    在每个 epoch 结束时:
      control.should_save = True
      control.should_evaluate = True
    => Trainer 会调用 _save_checkpoint() 并做 evaluation
    """
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        epoch_int = int(state.epoch)
        # 若要只在 epoch=12,24,36... 保存，可以改为 if epoch_int>0 and (epoch_int%12==0)
        if epoch_int > 0:
            control.should_save = True
            control.should_evaluate = True
        else:
            control.should_save = False
        return control


# ------------------------------------------------------
# 2) Callback: 每次验证后，打印2条示例输出
# ------------------------------------------------------
class EvalPrintSamplesCallback(TrainerCallback):
    """
    在 'on_evaluate' 阶段:
      - 从eval_dataset随机2条
      - 用 model.generate() 生成并打印
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

        eval_ds = trainer.eval_dataset
        if len(eval_ds) == 0:
            return

        sample_indices = random.sample(range(len(eval_ds)), min(2,len(eval_ds)))
        print("\n[EvalPrintSamplesCallback] Generating for 2 random samples from validation set:\n")

        for i in sample_indices:
            example = eval_ds[i]
            text_inp = example["qwen_formatted_text"]
            # 截取前N字符以防过长
            short_inp = text_inp[:300]
            if len(text_inp)>300:
                short_inp+="..."

            print(f"  [Sample idx={i}] input (partial): {short_inp}")

            # encode => 送入 model.generate
            inputs = tokenizer(
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
            print(f"  => generation: {gen_text}\n")


# ------------------------------------------------------
# 3) 加载 & 处理数据
# ------------------------------------------------------
def load_and_sample_dataset(dataset_name="AI-MO/numina_math-co_t",
                            sample_size=20000,
                            seed=42):
    ds = load_dataset(dataset_name)
    if "train" not in ds or "test" not in ds:
        raise ValueError("Dataset must have 'train' and 'test'. Found: %s" % list(ds.keys()))
    train_full = ds["train"]
    if sample_size>len(train_full):
        sample_size=len(train_full)
    train_sampled = train_full.shuffle(seed=seed).select(range(sample_size))
    return DatasetDict({"train": train_sampled, "test": ds["test"]})

def convert_to_qwen_chatml(example:Dict[str,Any])->Dict[str,str]:
    msgs = example.get("messages",[])
    if not msgs or not isinstance(msgs,list):
        return {"qwen_formatted_text":""}
    chat_str=""
    for turn in msgs:
        role = turn.get("role","user").lower()
        content = turn.get("content","").strip()
        if role=="assistant":
            chat_str+=f"<|im_start|>assistant {content}<|im_end|>\n"
        else:
            chat_str+=f"<|im_start|>user {content}<|im_end|>\n"
    return {"qwen_formatted_text":chat_str.strip()}

def split_train_val_test(dataset_dict:DatasetDict,val_ratio=0.1,seed=42)->DatasetDict:
    train_ds = dataset_dict["train"]
    splitted = train_ds.train_test_split(test_size=val_ratio,seed=seed)
    return DatasetDict({
        "train": splitted["train"],
        "validation": splitted["test"],
        "test": dataset_dict["test"]
    })

def save_dataset_as_json(dataset, file_path:str, text_col="qwen_formatted_text"):
    recs=[]
    for ex in dataset:
        recs.append({text_col: ex[text_col]})
    with open(file_path,"w",encoding="utf-8") as f:
        json.dump(recs,f,ensure_ascii=False,indent=2)


# ------------------------------------------------------
# 4) DataCollator & compute_metrics
# ------------------------------------------------------
@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer
    max_length:int=1024
    def __call__(self,features:List[Dict[str,Any]])->Dict[str,torch.Tensor]:
        texts=[ f["qwen_formatted_text"] for f in features ]
        enc=self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        enc["labels"] = enc["input_ids"].clone()
        return enc

def compute_metrics_for_lm(eval_pred):
    """
    - predictions => [batch, seq_len, vocab_size] logits
    - label_ids => [batch, seq_len]
    计算 token-level accuracy
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = logits.argmax(dim=-1)
    valid_mask = (labels!=-100)
    correct = (preds==labels)&valid_mask
    correct_count = correct.sum().item()
    total_count   = valid_mask.sum().item()
    accuracy = correct_count/max(total_count,1)
    return {"accuracy":accuracy}


# ------------------------------------------------------
# 5) LoRA微调函数
# ------------------------------------------------------
def train_qwen_lora(
    model_path:str,
    output_dir:str,
    train_dataset,
    val_dataset,
    num_train_epochs:int=4,
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
    print("[INFO] Loading base model + tokenizer from",model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # target modules
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"]
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM
    )
    print("[INFO] LoRA config:",config)

    model = get_peft_model(model,config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForCausalLM(tokenizer, max_seq_length)

    training_args = TrainingArguments(
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

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_for_lm,
        callbacks=callbacks  # 这里直接接收list[TrainerCallback]
    )

    print("[INFO] Start LoRA training ...")
    trainer.train()
    print("[INFO] Training done.")

    # final save
    print("[INFO] Save final LoRA model + tokenizer ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[INFO] Done, final model saved at:",output_dir)


# ------------------------------------------------------
# 6) main
# ------------------------------------------------------
def main():
    # 目录
    base_path = "/root/autodl-tmp/code/2025service_creativity/train_weight/numina_math_co_t"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_path,f"train_lora_{ts}")
    os.makedirs(output_dir,exist_ok=True)

    # 加载dataset
    ds_dict = load_and_sample_dataset(
        dataset_name="AI-MO/numina_math-co_t",
        sample_size=20000,
        seed=42
    )
    ds_dict = ds_dict.map(convert_to_qwen_chatml, desc="Convert to Qwen ChatML")
    ds_dict = split_train_val_test(ds_dict, val_ratio=0.1, seed=42)

    # 备份json
    save_dataset_as_json(ds_dict["train"],os.path.join(output_dir,"train.json"))
    save_dataset_as_json(ds_dict["validation"],os.path.join(output_dir,"val.json"))
    save_dataset_as_json(ds_dict["test"],os.path.join(output_dir,"test.json"))

    # 构建回调
    callback_epoch = SaveEveryEpochCallback()
    callback_eval_print = EvalPrintSamplesCallback()
    callbacks = [callback_epoch, callback_eval_print]

    # LoRA fine-tune
    model_path = "/root/autodl-tmp/model/Qwenmath_2.5_1.5b"
    train_qwen_lora(
        model_path=model_path,
        output_dir=output_dir,
        train_dataset=ds_dict["train"],
        val_dataset=ds_dict["validation"],
        num_train_epochs=3,    # demo
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
