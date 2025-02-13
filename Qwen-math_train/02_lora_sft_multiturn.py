#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_lora_sft_multiturn.py

对 "GoT + AutoCoT + 多轮对话" 数据进行LoRA微调，适配Qwen2-Math7B (GPT式)，
1) 每次运行会在 loraSFT/ 目录下自动创建子文件夹(按时间戳)，保存日志与模型输出
2) 调整 epoch/batch_size -> 避免过小
3) 修正 'EvalLoopOutput' 不支持 item assignment 的报错
   => 评估时直接从 eval_result.predictions / .label_ids 取预测/标签
4) evaluation_strategy => eval_strategy
"""

import os
import time
import json
import random
import logging
import torch
import numpy as np
from typing import List, Dict

from torch.utils.data import Dataset
from evaluate import load as load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


#########################################
# Step 0: Logging & Output Dir
#########################################
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
BASE_OUTDIR = "/root/autodl-tmp/code/2025service_creativity/Qwen-math_train/loraSFT"
RUN_OUTDIR  = os.path.join(BASE_OUTDIR, f"run_{TIMESTAMP}")
os.makedirs(RUN_OUTDIR, exist_ok=True)

log_file_path = os.path.join(RUN_OUTDIR, "training.log")

logging.basicConfig(
    filename=log_file_path,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger("").addHandler(console_handler)

logging.info("Starting LoRA SFT for Qwen-Math7B...")

#########################################
# Step 1: parse多轮对话
#########################################
def parse_conversation_data(conversations: List[Dict]) -> List[Dict]:
    data_samples = []
    for conv in conversations:
        conv_id = conv.get("conversation_id", "")
        msgs = conv.get("messages", [])
        context_so_far = []
        for i, msg in enumerate(msgs):
            role = msg.get("role","")
            content= msg.get("content","")
            if role=="assistant":
                # find user
                user_text = ""
                if i-1>=0 and msgs[i-1]["role"]=="user":
                    user_text = msgs[i-1]["content"]
                # gather GoT
                got_dict = msg.get("graph_of_thought", {})
                final_ans= msg.get("final_answer", "")
                try:
                    import json
                    got_str = json.dumps(got_dict, ensure_ascii=False)
                except:
                    got_str = str(got_dict)
                assistant_text = content + f"\n[GRAPH_OF_THOUGHT]\n{got_str}\n[FINAL_ANSWER]\n{final_ans}"

                combined_input = build_combined_input(context_so_far, user_text)
                data_samples.append({
                    "conversation_id": conv_id,
                    "context_so_far": combined_input,
                    "assistant_text": assistant_text
                })
            # add to context
            context_so_far.append((role, content))
    return data_samples

def build_combined_input(context_messages, user_text):
    system_prompt = "You are a helpful math assistant, capable of Graph-of-Thought reasoning.\n"
    lines = [f"SYSTEM: {system_prompt}"]
    for (r,c) in context_messages:
        lines.append(f"{r.upper()}: {c}")
    lines.append(f"USER: {user_text}")
    final_str = "\n".join(lines)
    return final_str

#########################################
# Step 2: dataset & collator
#########################################
class MultiTurnGoTDataset(Dataset):
    def __init__(self, data_list, tokenizer):
        self.data = data_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        conv_id = sample["conversation_id"]
        context_str   = sample["context_so_far"]
        assistant_str = sample["assistant_text"]

        # final
        full_text = context_str + "\nASSISTANT:\n" + assistant_str
        enc = self.tokenizer(full_text, truncation=True, max_length=1024)
        input_ids = enc["input_ids"]
        labels = input_ids.copy()

        # identify assistant offset
        context_enc = self.tokenizer(context_str + "\nASSISTANT:\n", truncation=True, max_length=1024)
        assistant_offset= len(context_enc["input_ids"])
        for i in range(assistant_offset):
            labels[i] = -100
        return {
            "input_ids": input_ids,
            "labels": labels
        }

def data_collator(features:List[Dict]):
    # find max length
    batch_input_ids = [f["input_ids"] for f in features]
    batch_labels    = [f["labels"]    for f in features]
    max_len = max(len(ids) for ids in batch_input_ids)

    input_ids_padded = []
    labels_padded    = []
    attention_masks  = []
    for i in range(len(batch_input_ids)):
        ids = batch_input_ids[i]
        lab = batch_labels[i]
        pad_len = max_len - len(ids)
        ids_padded = ids + [self_pad_token_id]*pad_len
        lab_padded = lab + [-100]*pad_len
        attn_mask  = [1]*len(ids) + [0]*pad_len

        input_ids_padded.append(ids_padded)
        labels_padded.append(lab_padded)
        attention_masks.append(attn_mask)

    import torch
    input_ids_pt     = torch.tensor(input_ids_padded, dtype=torch.long)
    labels_pt        = torch.tensor(labels_padded,    dtype=torch.long)
    attention_mask_pt= torch.tensor(attention_masks,  dtype=torch.long)

    return {
        "input_ids": input_ids_pt,
        "labels": labels_pt,
        "attention_mask": attention_mask_pt
    }

#########################################
# Step 3: Custom Trainer
#########################################
class CausalLMTrainer(Trainer):
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer=tokenizer

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        loss, _, _ = super().prediction_step(
            model, inputs, prediction_loss_only=True, ignore_keys=ignore_keys
        )
        if prediction_loss_only:
            return (loss, None, None)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        gen_kwargs = dict(
            max_new_tokens=1024,
            num_beams=2,
            do_sample=False,
            no_repeat_ngram_size=2
        )
        generated = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            **gen_kwargs
        )
        batch_size= input_ids.shape[0]
        preds=[]
        for i in range(batch_size):
            prompt_len= attention_mask[i].sum().item()
            gen_ids= generated[i][prompt_len:]
            pred_str= self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            preds.append(pred_str)
        label_ids= inputs["labels"]
        label_ids= torch.where(label_ids==-100, torch.zeros_like(label_ids), label_ids)
        refs=[]
        for i in range(batch_size):
            ref_str= self.tokenizer.decode(label_ids[i], skip_special_tokens=True)
            refs.append(ref_str)
        return (loss, preds, refs)

    # NOTE: we do NOT do out["eval_preds"] in eval loop to avoid TypeError
    # We'll just rely on the returned namedtuple: predictions, label_ids, metrics, etc.

#########################################
# Step 4: compute_metrics
#########################################
from evaluate import load as load_metric
def compute_metrics(pred_texts, ref_texts):
    rouge = load_metric("rouge")
    result= rouge.compute(predictions=pred_texts, references=ref_texts)
    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"]
    }

#########################################
# Step 5: main
#########################################
def main():
    import logging
    try:
        data_dir= "/root/autodl-tmp/code/2025service_creativity/Qwen-math_train/ConvGoTAzure99_blossom-math-v1"
        train_json= os.path.join(data_dir,"train.json")
        val_json=   os.path.join(data_dir,"val.json")
        test_json=  os.path.join(data_dir,"test.json")

        def load_convs(path):
            with open(path,"r",encoding="utf-8") as f:
                return json.load(f)

        train_convs= load_convs(train_json)
        val_convs  = load_convs(val_json)
        test_convs = load_convs(test_json)
        train_samples= parse_conversation_data(train_convs)
        val_samples=   parse_conversation_data(val_convs)
        test_samples=  parse_conversation_data(test_convs)
        logging.info(f"[train] {len(train_samples)}, [val] {len(val_samples)}, [test] {len(test_samples)}")

        # create new subdir for run
        from datetime import datetime
        subrun_name= datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir= os.path.join(BASE_OUTDIR, subrun_name)
        os.makedirs(run_dir, exist_ok=True)

        # load Qwen
        model_path= "/root/autodl-tmp/model/Qwen2-math7B"
        tokenizer= AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token= tokenizer.eos_token
        global self_pad_token_id
        self_pad_token_id= tokenizer.pad_token_id

        base_model= AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        # LoRA
        lora_cfg= LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model= get_peft_model(base_model, lora_cfg)

        # build dataset
        train_ds= MultiTurnGoTDataset(train_samples, tokenizer)
        val_ds=   MultiTurnGoTDataset(val_samples, tokenizer)
        test_ds=  MultiTurnGoTDataset(test_samples, tokenizer)

        from transformers import TrainingArguments
        training_args= TrainingArguments(
            output_dir= run_dir,   # store ckpt in new run dir
            num_train_epochs=100,    # can increase
            per_device_train_batch_size=2,  # bigger than 1 for efficiency
            per_device_eval_batch_size=2,
            eval_steps=250,        # do eval less frequently
            save_steps=2500,
            logging_steps=2500,
            learning_rate=5e-5,    # bigger LR if data large
            fp16=True,
            overwrite_output_dir=True,
            remove_unused_columns=False,
            eval_strategy="steps",  # replaced evaluation_strategy
        )

        trainer= CausalLMTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator= data_collator
        )

        logging.info("Begin training LoRA SFT ...")
        trainer.train()
        logging.info("Training done.")

        # save lora
        lora_out= os.path.join(run_dir,"lora_weight")
        model.save_pretrained(lora_out)
        logging.info(f"LoRA saved => {lora_out}")

        # evaluate val
        val_result= trainer.evaluate(eval_dataset=val_ds)
        # we can retrieve predictions from val_result.predictions
        val_preds_torch= val_result.predictions
        val_labelids_torch= val_result.label_ids
        # decode
        val_preds_str= []
        val_refs_str= []
        for i in range(len(val_preds_torch)):
            # remove prompt
            pred_ids= val_preds_torch[i]
            # these are not parted by prompt_len in batched scenario => we just decode all
            # if you want to remove prompt portion => you'd need to store input lengths
            pred_str= tokenizer.decode(pred_ids, skip_special_tokens=True)
            val_preds_str.append(pred_str)
            # label
            lab_ids= val_labelids_torch[i]
            lab_ids= np.where(lab_ids<0, 0, lab_ids)
            lab_str= tokenizer.decode(lab_ids, skip_special_tokens=True)
            val_refs_str.append(lab_str)

        val_rouge= compute_metrics(val_preds_str, val_refs_str)
        logging.info(f"[Val ROUGE] => {val_rouge}")

        # evaluate test
        test_result= trainer.evaluate(eval_dataset=test_ds)
        test_preds_torch= test_result.predictions
        test_labelids_torch= test_result.label_ids
        test_preds_str= []
        test_refs_str= []
        for i in range(len(test_preds_torch)):
            pred_str= tokenizer.decode(test_preds_torch[i], skip_special_tokens=True)
            test_preds_str.append(pred_str)
            lab_ids= test_labelids_torch[i]
            lab_ids= np.where(lab_ids<0, 0, lab_ids)
            lab_str= tokenizer.decode(lab_ids, skip_special_tokens=True)
            test_refs_str.append(lab_str)

        test_rouge= compute_metrics(test_preds_str, test_refs_str)
        logging.info(f"[Test ROUGE] => {test_rouge}")

        logging.info("All done. results in => "+run_dir)

    except Exception as e:
        logging.exception(f"Error during LoRA SFT: {e}")
        raise

if __name__=="__main__":
    main()
