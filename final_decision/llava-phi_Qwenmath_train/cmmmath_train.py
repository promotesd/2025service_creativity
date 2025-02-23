#!/usr/bin/env python
# coding: utf-8

import os
import json
import time
import torch
import logging
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    LlavaForConditionalGeneration,
    AutoModelForCausalLM
)
from tqdm import tqdm

# 引入您已有的多模态桥接脚本
from qformer_llava_Qwenmath_model import (
    LlavaPhiVisionEncoder,
    QFormer,
    VisionAggregator,
    AdvancedMultiModalQwen
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def to_str(obj):
    """
    将 obj 转换为字符串:
     - 如果是列表, 用空格或换行拼接
     - 如果是其他非字符串类型, 用 str() 转换
    """
    if isinstance(obj, list):
        # 比如选项是 ["选项A", "选项B"] => "选项A\n选项B"
        return "\n".join(str(x) for x in obj)
    else:
        return str(obj)


# ---------------------------
# 1) 多图Dataset
# ---------------------------
class CmmMathMultiImageDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        image_dir: str,
        tokenizer,
        processor,
        max_text_length=256,
        split="train"
    ):
        self.data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_text_length = max_text_length
        self.split = split

        logger.info(f"[{split}] Loaded {len(self.data)} samples from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 转成字符串
        question_text = to_str(item.get("question", ""))
        options_text  = to_str(item.get("options", ""))
        answer_text   = to_str(item.get("answer", ""))

        # 替换 <ImageHere> => <Image>
        question_text = question_text.replace("<ImageHere>", "<Image>")
        options_text  = options_text.replace("<ImageHere>", "<Image>")

        # 构造训练Prompt
        prompt_text = (
            "你是一名数学老师。请阅读题目并选出正确答案。\n"
            f"题目：{question_text}\n"
            f"选项：{options_text}\n"
            "回答："
        )
        label_text = answer_text
        full_text = prompt_text + label_text

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        labels = input_ids.clone()

        # 处理多张图片
        img_list = item.get("image", [])
        pixel_values_list = []
        if len(img_list) == 0:
            pixel_values_list.append(self._get_empty_image())
        else:
            for fn in img_list:
                img_path = os.path.join(self.image_dir, fn)
                if not os.path.isfile(img_path):
                    pixel_values_list.append(self._get_empty_image())
                else:
                    pil_img = Image.open(img_path).convert("RGB")
                    img_enc = self.processor(pil_img, text="", return_tensors="pt")
                    pixel_values_list.append(img_enc["pixel_values"][0])

        return {
            "prompt_text": prompt_text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values_list": pixel_values_list
        }

    def _get_empty_image(self, size=(224, 224)):
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img = Image.fromarray(arr, 'RGB')
        img_enc = self.processor(img, text="", return_tensors="pt")
        return img_enc["pixel_values"][0]


def multi_image_collate_fn(batch):
    """
    将Dataset返回的list of dict合并成batch.
    pixel_values_list是多图 => [batch_size, list_of_images(T), each [3,H,W]]
    """
    prompt_texts = [item["prompt_text"] for item in batch]
    input_ids_list = [item["input_ids"] for item in batch]
    attention_mask_list = [item["attention_mask"] for item in batch]
    labels_list = [item["labels"] for item in batch]
    pixel_values_lists = [item["pixel_values_list"] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask_list, batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=-100
    )

    return {
        "prompt_texts": prompt_texts,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values_lists": pixel_values_lists
    }


# ---------------------------
# 2) 改造模型以处理多图 aggregator + Padding
# ---------------------------
class AdvancedMultiModalQwenMultiImage(torch.nn.Module):
    """
    1) vision_encoder + aggregator => aggregator tokens
       (对多张图分别encoder/aggregator, 然后cat)
    2) 因为每个样本多张图数量不一定相同，所以 aggregator_tokens 长度会不同。
       我们对 batch 内各样本做 Padding，让 (batch_size, max_len, hidden_dim) 对齐。
       用 vision_mask 标记有效部分。
    3) Q-Former => (t_fused, v_fused)
    4) Qwen => outputs
    """

    def __init__(
        self,
        llava_vision_encoder: LlavaPhiVisionEncoder,
        qwen_model: AutoModelForCausalLM,
        aggregator_token_count: int = 8,
        in_dim: int = 1024,
        qformer_hidden_dim: int = 2048,
        aggregator_heads: int = 8,
        qformer_layers: int = 4,
        qformer_heads: int = 8,
        use_mrope: bool = True
    ):
        super().__init__()
        self.vision_encoder = llava_vision_encoder
        self.qwen_model = qwen_model

        self.aggregator = VisionAggregator(
            in_dim=in_dim,
            out_dim=qformer_hidden_dim,
            aggregator_token_count=aggregator_token_count,
            num_heads=aggregator_heads
        )

        self.qformer = QFormer(
            hidden_dim=qformer_hidden_dim,
            num_heads=qformer_heads,
            num_layers=qformer_layers,
            use_mrope=use_mrope
        )

        self.aggregator_token_count = aggregator_token_count

    def forward(
        self,
        pixel_values_lists,  # list of lists [ [img_t1, img_t2, ...], [img_t1,...], ...]
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None
    ):
        device = input_ids.device

        # A) 对每条样本多图 => aggregator
        bridging_feats_list = []  # 保存每条样本的 aggregator feats (shape=[1, #tokens, hidden_dim])
        len_list = []             # 保存每条样本 aggregator token 数量
        for pixel_list in pixel_values_lists:
            # 对一条样本
            # 依次处理多张图 => cat
            sample_agg = []
            for img_t in pixel_list:
                img_t = img_t.unsqueeze(0).to(device)
                # vision encoder => [1, seq, 1024]
                with torch.no_grad():
                    feats = self.vision_encoder(img_t)
                aggregator_dtype = self.aggregator.vis_proj.weight.dtype
                feats = feats.to(aggregator_dtype)
                bridging_feats = self.aggregator(feats)  # [1, aggregator_token_count, hidden_dim]
                sample_agg.append(bridging_feats)

            # cat在 seq维度=1
            sample_agg_cat = torch.cat(sample_agg, dim=1)  # [1, aggregator_token_count * T, hidden_dim]
            bridging_feats_list.append(sample_agg_cat)
            len_list.append(sample_agg_cat.shape[1])  # aggregator token数

        # B) Padding: 对 batch 内 aggregator tokens 不同长度做对齐
        bridging_feats_batch, vision_mask = self._pad_aggregator_feats_for_batch(
            bridging_feats_list, len_list
        )  # => shape [B, max_len, hidden_dim], [B, max_len]
        # bridging_feats_batch, vision_mask 都在 aggregator's dtype / device

        # C) 构造 text embeds
        text_embeds = self.qwen_model.get_input_embeddings()(input_ids).to(bridging_feats_batch.dtype)

        # Q-Former
        t_fused, v_fused = self.qformer(
            text_embeds, bridging_feats_batch,
            text_mask=attention_mask.bool() if attention_mask is not None else None,
            vision_mask=vision_mask
        )

        # Qwen decode
        qwen_dtype = self.qwen_model.get_input_embeddings().weight.dtype
        t_fused = t_fused.to(qwen_dtype)

        outputs = self.qwen_model(
            inputs_embeds=t_fused,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def _pad_aggregator_feats_for_batch(self, bridging_feats_list, len_list):
        """
        # 关键改动:
        bridging_feats_list: list of Tensors, each shape [1, aggregator_tokens_i, hidden_dim]
        len_list: aggregator_tokens_i for each sample
        return bridging_feats_batch: [B, max_len, hidden_dim]
        and vision_mask: [B, max_len], True表示有效
        """
        device = bridging_feats_list[0].device
        dtype = bridging_feats_list[0].dtype
        B = len(bridging_feats_list)
        max_len = max(len_list)
        hidden_dim = bridging_feats_list[0].shape[-1]

        bridging_feats_batch = torch.zeros(B, max_len, hidden_dim, dtype=dtype, device=device)
        vision_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)

        for i in range(B):
            cur_len = len_list[i]
            bridging_feats_batch[i, :cur_len, :] = bridging_feats_list[i][0, :cur_len, :]
            vision_mask[i, :cur_len] = True

        return bridging_feats_batch, vision_mask

    @torch.no_grad()
    def generate(
        self,
        pixel_values_lists,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        **gen_kwargs
    ):
        device = prompt_ids.device

        bridging_feats_list = []
        len_list = []
        for pixel_list in pixel_values_lists:
            sample_agg = []
            for img_t in pixel_list:
                img_t = img_t.unsqueeze(0).to(device)
                with torch.no_grad():
                    feats = self.vision_encoder(img_t)
                aggregator_dtype = self.aggregator.vis_proj.weight.dtype
                feats = feats.to(aggregator_dtype)
                bridging_feats = self.aggregator(feats)
                sample_agg.append(bridging_feats)
            sample_agg_cat = torch.cat(sample_agg, dim=1)
            bridging_feats_list.append(sample_agg_cat)
            len_list.append(sample_agg_cat.shape[1])

        bridging_feats_batch, vision_mask = self._pad_aggregator_feats_for_batch(bridging_feats_list, len_list)

        text_embeds = self.qwen_model.get_input_embeddings()(prompt_ids).to(bridging_feats_batch.dtype)

        t_fused, v_fused = self.qformer(
            text_embeds, bridging_feats_batch,
            text_mask=attention_mask.bool() if attention_mask is not None else None,
            vision_mask=vision_mask
        )

        qwen_dtype = self.qwen_model.get_input_embeddings().weight.dtype
        t_fused = t_fused.to(qwen_dtype)

        gen_out = self.qwen_model.generate(
            inputs_embeds=t_fused,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **gen_kwargs
        )
        return gen_out


# ---------------------------
# 3) 训练逻辑
# ---------------------------
def evaluate_model(model, tokenizer, dataset, device, max_eval_samples=2):
    from torch.utils.data import Subset
    if max_eval_samples <= 0:
        return 0.0
    if max_eval_samples > len(dataset):
        max_eval_samples = len(dataset)

    indices = random.sample(range(len(dataset)), max_eval_samples)
    subset_data = Subset(dataset, indices)
    loader = DataLoader(subset_data, batch_size=1, collate_fn=multi_image_collate_fn)

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pixel_values_lists = batch["pixel_values_lists"]

            outputs = model(
                pixel_values_lists=pixel_values_lists,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()

            generated_ids = model.generate(
                pixel_values_lists=pixel_values_lists,
                prompt_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50
            )
            gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            print(f"\n=== [EVAL SAMPLE {step}] ===")
            print("[Prompt Text]:", batch["prompt_texts"][0])
            print("[Model Output]:", gen_text)
            print("===========================")

    avg_loss = total_loss / max_eval_samples
    model.train()
    return avg_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 数据路径
    cmm_root = "/root/autodl-tmp/dataset/cmm-math"
    train_data_file = os.path.join(cmm_root, "train_data.jsonl")
    train_image_dir = os.path.join(cmm_root, "images", "Train_Images")

    llava_model_path = "/root/autodl-tmp/model/llava-phi-3-mini-hf"
    qwenmath_path = "/root/autodl-tmp/model/Qwenmath_2.5_1.5b"

    # 2) 创建保存目录
    base_save_dir = "/root/autodl-tmp/code/2025service_creativity/train_weight/cmmmath_train"
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    run_save_dir = os.path.join(base_save_dir, f"cmmmath_train_{time_str}")
    os.makedirs(run_save_dir, exist_ok=True)
    logger.info(f"Created run folder: {run_save_dir}")

    # 3) 加载模型
    logger.info("Loading LLaVA-Phi...")
    llava_full = LlavaForConditionalGeneration.from_pretrained(
        llava_model_path,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32
    ).eval().to(device)

    llava_vision_encoder = LlavaPhiVisionEncoder(llava_full)
    for p in llava_vision_encoder.parameters():
        p.requires_grad = False

    logger.info("Loading Qwen...")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        qwenmath_path,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32
    ).eval().to(device)
    for p in qwen_model.parameters():
        p.requires_grad = False

    logger.info("Building multi-image bridging model...")
    multi_model = AdvancedMultiModalQwenMultiImage(
        llava_vision_encoder=llava_vision_encoder,
        qwen_model=qwen_model,
        aggregator_token_count=8,
        in_dim=1024,
        qformer_hidden_dim=1536,
        aggregator_heads=8,
        qformer_layers=4,
        qformer_heads=8,
        use_mrope=True
    ).to(device)

    # 只训练 aggregator + qformer
    trainable_params = []
    if multi_model.aggregator.query_tokens is not None:
        trainable_params.append(multi_model.aggregator.query_tokens)
        trainable_params += list(multi_model.aggregator.vis_proj.parameters())
        trainable_params += list(multi_model.aggregator.norm_vis.parameters())
        trainable_params += list(multi_model.aggregator.norm_q.parameters())
    trainable_params += list(multi_model.qformer.parameters())

    logger.info("Loading tokenizer & processor...")
    tokenizer = AutoTokenizer.from_pretrained(qwenmath_path, use_fast=False)
    processor = AutoProcessor.from_pretrained(llava_model_path)

    # 4) 数据集 & DataLoader
    from torch.utils.data import DataLoader
    logger.info("Loading multi-image dataset...")
    train_dataset = CmmMathMultiImageDataset(
        data_file=train_data_file,
        image_dir=train_image_dir,
        tokenizer=tokenizer,
        processor=processor,
        max_text_length=256,
        split="train"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=multi_image_collate_fn
    )

    # 5) 优化器
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    scaler = torch.cuda.amp.GradScaler() if device=="cuda" else None

    # 6) 训练循环
    num_epochs = 36
    multi_model.train()
    loss_log_path = os.path.join(run_save_dir, "loss_log.txt")

    logger.info(f"Start training for {num_epochs} epochs...")

    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")

        for step, batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pixel_values_lists = batch["pixel_values_lists"]  # list of lists

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = multi_model(
                        pixel_values_lists=pixel_values_lists,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = multi_model(
                    pixel_values_lists=pixel_values_lists,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            global_step += 1

            current_loss = total_loss / (step + 1)
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

        epoch_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} completed. Average loss={epoch_loss:.4f}")

        with open(loss_log_path, "a", encoding="utf-8") as f:
            f.write(f"Epoch {epoch+1}, Average loss={epoch_loss:.4f}\n")

        # evaluation
        eval_out_path = os.path.join(run_save_dir, f"evaluation_epoch_{epoch+1}.txt")
        eval_loss = evaluate_model(
            model=multi_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            device=device,
            max_eval_samples=200
        )
        logger.info(f"Evaluation (epoch {epoch+1}): average eval loss={eval_loss:.4f}")

        with open(eval_out_path, "w", encoding="utf-8") as f:
            f.write(f"Epoch {epoch+1} - Evaluation average loss={eval_loss:.4f}\n")
            f.write("请查看日志中[Prompt Text]和[Model Output]。\n")

        # 保存 checkpoint
        epoch_ckpt_path = os.path.join(run_save_dir, f"epoch_{epoch+1}.pth")
        torch.save({
            "aggregator": multi_model.aggregator.state_dict(),
            "qformer": multi_model.qformer.state_dict()
        }, epoch_ckpt_path)
        logger.info(f"[Epoch {epoch+1}] saved to {epoch_ckpt_path}")


if __name__ == "__main__":
    main()
