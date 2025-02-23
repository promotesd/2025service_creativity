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
# A) 子回调1：每个整epoch保存+验证
# ------------------------------------------------
class SaveEveryEpochCallback(TrainerCallback):
    """
    在每个整 epoch (epoch>=1)结束时:
      - should_save = True
      - should_evaluate = True
    这样 Trainer 会在内部 `_save_checkpoint()` 并自动调用 evaluate()
    如果只想每12 epoch 保存:
      在 if epoch_int>0: => 改 if (epoch_int>0 and epoch_int%12==0):
    """
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_int = int(state.epoch)
        if epoch_int>0:
            control.should_save=True
            control.should_evaluate=True
        else:
            control.should_save=False
        return control


# ------------------------------------------------
# B) 子回调2：验证后随机打印2条输出
# ------------------------------------------------
class EvalPrintSamplesCallback(TrainerCallback):
    """
    在 on_evaluate 阶段:
      - 随机抽2条验证集
      - model.generate => 打印
    """
    def on_evaluate(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if model is None or tokenizer is None:
            return
        trainer = kwargs.get("trainer", None)
        if trainer is None:
            return

        val_ds = trainer.eval_dataset
        if not val_ds or len(val_ds)==0:
            return

        sample_idxes= random.sample(range(len(val_ds)), min(2, len(val_ds)))
        print("\n[EvalPrintSamplesCallback] sample 2 from val:\n")
        for idx in sample_idxes:
            ex= val_ds[idx]
            text_inp= ex["qwen_formatted_text"]
            short_inp= text_inp[:300]
            if len(text_inp)>300: short_inp+="..."
            print(f" [val idx={idx}] partial input: {short_inp}")

            inputs= tokenizer(
                text_inp,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            for k in inputs:
                inputs[k]=inputs[k].to(model.device)

            gen_out= model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                top_p=0.9,
                temperature=0.8
            )
            gen_text= tokenizer.decode(gen_out[0], skip_special_tokens=True)
            print(f" => generation:\n{gen_text}\n")


# ------------------------------------------------
# C) 合并回调 => CompositeCallback
# ------------------------------------------------
class CompositeCallback(TrainerCallback):
    """
    接受多个回调实例, 并在 Trainer各事件发生时, 依次调用子回调.
    用于兼容老版本 Transformers不支持 [cb1, cb2].
    """
    def __init__(self, callbacks: List[TrainerCallback]):
        self.callbacks= callbacks or []

    def on_init_end(self, args, state, control, **kwargs):
        for cb in self.callbacks:
            control = cb.on_init_end(args, state, control, **kwargs)
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        for cb in self.callbacks:
            control = cb.on_train_begin(args, state, control, **kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        for cb in self.callbacks:
            control = cb.on_train_end(args, state, control, **kwargs)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        for cb in self.callbacks:
            control = cb.on_epoch_end(args, state, control, **kwargs)
        return control

    def on_evaluate(self, args, state, control, model=None, tokenizer=None, **kwargs):
        for cb in self.callbacks:
            control = cb.on_evaluate(args, state, control, model=model, tokenizer=tokenizer, **kwargs)
        return control

    # 你可根据需求把 trainercallback 的其余 hook 也写上,
    # 如 on_step_end, on_log, on_save等, 这里只演示常用


# ------------------------------------------------
# D) 数据加载&处理
# ------------------------------------------------
def load_and_keep_relevant_fields() -> DatasetDict:
    """
    从 'eth-nlped/mathdial' 加载 => {train(2262), test(599)}
    保留 question, ground_truth, conversation
    """
    from datasets import load_dataset
    ds = load_dataset("eth-nlped/mathdial")
    # ds => {train, test}
    columns_to_remove=[
        "qid","scenario","student_incorrect_solution","student_profile","teacher_described_confusion",
        "self-correctness","self-typical-confusion","self-typical-interactions"
    ]
    def _select_cols(ex):
        return {
            "question": ex["question"],
            "ground_truth": ex["ground_truth"],
            "conversation": ex["conversation"]
        }
    ds=ds.map(_select_cols, remove_columns=columns_to_remove)
    return ds

def parse_conversation_string(conv_str: str)->List[Dict[str,str]]:
    if not isinstance(conv_str,str) or not conv_str.strip():
        return []
    segments=[seg.strip() for seg in conv_str.split("|EOM|") if seg.strip()]
    turns=[]
    for seg in segments:
        if ":" in seg:
            role_part, text_part= seg.split(":",1)
            role= role_part.strip()
            text= text_part.strip()
        else:
            role="Unknown"
            text= seg.strip()
        turns.append({"role": role, "text": text})
    return turns

def convert_to_qwen_chatml(
    example:Dict[str,Any],
    insert_got=True,
    insert_auto_cot=True,
    insert_self_reflection=True
)->Dict[str,str]:
    """
    将 question + conversation => Qwen ChatML
    注入 GoT, AutoCoT, 自反思
    """
    question= example.get("question","").strip()
    ground_truth= example.get("ground_truth","").strip()
    conv_str= example.get("conversation","")

    # 解析多轮
    turns= parse_conversation_string(conv_str)
    # 在对话前插入 question
    if question:
        turns.insert(0, {"role":"UserQuestion","text": question})

    dialogue_str=""
    for i,turn in enumerate(turns):
        role_lower= turn["role"].lower()
        text= turn["text"].strip()

        if role_lower=="teacher":
            # => assistant
            # GoT提示
            if i==1 and insert_got:
                text= "[GoT提示]: 让我们先审题并逐步推理.\n" + text
            # 最后teacher => 自反思 & chain-of-thought
            if i==len(turns)-1:
                if insert_self_reflection:
                    text+="\n[Self-Reflection]: 再次检查是否有疏漏."
                if insert_auto_cot and ground_truth:
                    text+=f"\n[Chain-of-Thought]: {ground_truth}"

            dialogue_str += f"<|im_start|>assistant {text}<|im_end|>\n"
        else:
            # => user
            dialogue_str += f"<|im_start|>user {text}<|im_end|>\n"

    return {"qwen_formatted_text": dialogue_str.strip()}

def split_dataset(ds:DatasetDict, val_ratio=0.1, seed=42)->DatasetDict:
    train_ds= ds["train"]
    test_ds= ds["test"]
    splitted= train_ds.train_test_split(test_size=val_ratio, seed=seed)
    return DatasetDict({
        "train": splitted["train"],
        "validation": splitted["test"],
        "test": test_ds
    })

def save_dataset_as_json(dataset, file_path:str, text_col="qwen_formatted_text"):
    recs=[]
    for ex in dataset:
        recs.append({text_col: ex[text_col]})
    with open(file_path,"w",encoding="utf-8") as f:
        json.dump(recs,f,ensure_ascii=False,indent=2)


# ------------------------------------------------
# E) Collator + metric
# ------------------------------------------------
@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer
    max_length:int=1024

    def __call__(self, features:List[Dict[str,Any]])->Dict[str,torch.Tensor]:
        texts=[ f["qwen_formatted_text"] for f in features ]
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
    => token-level accuracy
    """
    logits, labels= eval_pred.predictions, eval_pred.label_ids
    preds= logits.argmax(dim=-1)
    valid_mask=(labels!=-100)
    correct=((preds==labels)&valid_mask).sum().item()
    total= valid_mask.sum().item()
    accuracy= correct/max(total,1)
    return {"accuracy":accuracy}


# ------------------------------------------------
# F) LoRA微调
# ------------------------------------------------
def train_qwen_mathdial_lora(
    model_path:str,
    output_dir:str,
    train_dataset,
    val_dataset,
    num_train_epochs:int=32,
    eval_each_epoch=True,
    callbacks=None,
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    max_seq_length=1024,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.05
):
    tokenizer= AutoTokenizer.from_pretrained(model_path, use_fast=False)
    base_model= AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # LoRA
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

    collator= DataCollatorForCausalLM(tokenizer, max_seq_length)

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
        save_strategy="no", 
        save_total_limit=3,
        remove_unused_columns=False
    )
    from transformers import Trainer

    trainer= Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics_for_lm,
        callbacks=[callbacks] if callbacks else None  # 传入 "单个" callback
    )

    print("[INFO] Start LoRA training on mathdial ...")
    trainer.train()
    print("[INFO] Training done")

    print("[INFO] Save final LoRA model & tokenizer ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[INFO] Done. LoRA model saved at:", output_dir)


# ------------------------------------------------
# G) 主函数
# ------------------------------------------------
def main():
    base_path= "/root/autodl-tmp/code/2025service_creativity/train_weight/mathdial_Qwen_lora"
    ts= datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir= os.path.join(base_path,f"train_lora_{ts}")
    os.makedirs(output_dir,exist_ok=True)

    # 1) 加载 mathdial => question, ground_truth, conversation
    ds= load_and_keep_relevant_fields()  # {train(2262), test(599)}

    # 2) 转Qwen ChatML(GoT+AutoCoT+自反思)
    ds= ds.map(
        lambda ex: convert_to_qwen_chatml(ex, True, True, True),
        desc="Converting to Qwen ChatML..."
    )

    # 3) 拆分 => train(90%), val(10%), test保留
    ds= split_dataset(ds,val_ratio=0.1, seed=42)
    # => {train(2035?), validation(227?), test(599)}

    # 4) dump
    save_dataset_as_json(ds["train"], os.path.join(output_dir,"mathdial_train.json"))
    save_dataset_as_json(ds["validation"], os.path.join(output_dir,"mathdial_validation.json"))
    save_dataset_as_json(ds["test"], os.path.join(output_dir,"mathdial_test.json"))

    # 5) 构造回调
    save_cb= SaveEveryEpochCallback()  # or SaveEvery12EpochsCallback
    eval_cb= EvalPrintSamplesCallback()
    # 我们需要把它们合并为一个callback => CompositeCallback
    # 这样只传1个callback给Trainer
    class CompositeCallback(TrainerCallback):
        def __init__(self, cbs):
            self.cbs=cbs
        def on_init_end(self, args, state, control, **kwargs):
            for cb in self.cbs:
                control= cb.on_init_end(args, state, control, **kwargs)
            return control
        def on_train_begin(self, args, state, control, **kwargs):
            for cb in self.cbs:
                control= cb.on_train_begin(args, state, control, **kwargs)
            return control
        def on_epoch_end(self, args, state, control, **kwargs):
            for cb in self.cbs:
                control= cb.on_epoch_end(args, state, control, **kwargs)
            return control
        def on_evaluate(self, args, state, control, model=None, tokenizer=None, **kwargs):
            for cb in self.cbs:
                control= cb.on_evaluate(args, state, control, model=model, tokenizer=tokenizer, **kwargs)
            return control
        def on_train_end(self, args, state, control, **kwargs):
            for cb in self.cbs:
                control= cb.on_train_end(args, state, control, **kwargs)
            return control
        # (可以继续添加别的回调Hook, 这里只演示常用)

    composite_cb= CompositeCallback([save_cb, eval_cb])

    # 6) LoRA
    model_path="/root/autodl-tmp/model/Qwenmath_2.5_1.5b"
    train_ds= ds["train"]
    val_ds= ds["validation"]

    train_qwen_mathdial_lora(
        model_path=model_path,
        output_dir=output_dir,
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_train_epochs=32,  # for example
        eval_each_epoch=True,
        callbacks=composite_cb,  # 只传一个callback(CompositeCallback)
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
