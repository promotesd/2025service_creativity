#!/usr/bin/env python
# coding: utf-8

"""
示例：将 llava-phi-3-mini-hf (视觉+bridging) 与 QwenMath_2.5_1.5b 语言解码拼接
并修正：1) vision_tower返回的是BaseModelOutput, 需提取 .last_hidden_state
       2) processor需显式设置 patch_size & vision_feature_select_strategy
"""

import torch
import torch.nn as nn
from PIL import Image
import requests

# pip install transformers accelerate safetensors sentencepiece

from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer
)


class LlavaPhiVisionEncoder(nn.Module):
    def __init__(self, llava_model: LlavaForConditionalGeneration):
        super().__init__()
        # 1) 获取 vision_tower
        vision_tower = getattr(llava_model, "vision_tower", None)
        if vision_tower is None and hasattr(llava_model, "model"):
            vision_tower = getattr(llava_model.model, "vision_tower", None)
        if vision_tower is None:
            raise AttributeError("无法在 llava_model 中找到 vision_tower。")

        if isinstance(vision_tower, (list, nn.ModuleList)):
            if len(vision_tower) == 0:
                raise ValueError("llava_model.vision_tower 是空。")
            self.vision_tower = vision_tower[0]
        else:
            self.vision_tower = vision_tower

        # 2) 获取 bridging
        visual_proj = None
        for cname in ["multi_modal_projector", "visual_projection"]:
            if hasattr(llava_model, cname):
                visual_proj = getattr(llava_model, cname)
                break
        if visual_proj is None and hasattr(llava_model, "model"):
            if hasattr(llava_model.model, "multi_modal_projector"):
                visual_proj = getattr(llava_model.model, "multi_modal_projector")
            elif hasattr(llava_model.model, "visual_projection"):
                visual_proj = getattr(llava_model.model, "visual_projection")

        if visual_proj is None:
            raise AttributeError("无法在 llava_model 中找到 bridging (multi_modal_projector / visual_projection).")

        self.visual_proj = visual_proj

    def forward(self, images: torch.Tensor):
        # vision_tower => 返回的是 CLIPVisionModelOutput / BaseModelOutputWithPooling
        # 需要取 .last_hidden_state
        outputs = self.vision_tower(images)  # e.g. CLIPVisionModelOutput
        vision_feats = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        # bridging => [batch, seq_len, bridging_dim]
        projected = self.visual_proj(vision_feats)
        return projected


class MultiModalQwen(nn.Module):
    def __init__(self, llava_vision_encoder: nn.Module, qwen_model: nn.Module):
        super().__init__()
        self.vision_encoder = llava_vision_encoder
        self.qwen_model = qwen_model

        # bridging dim
        bridging_dim = None
        if hasattr(self.vision_encoder.visual_proj, "linear_2"):
            bridging_dim = self.vision_encoder.visual_proj.linear_2.out_features
        elif hasattr(self.vision_encoder.visual_proj, "out_features"):
            bridging_dim = self.vision_encoder.visual_proj.out_features

        qwen_hidden = getattr(qwen_model.config, "hidden_size", None)

        if bridging_dim and qwen_hidden:
            if bridging_dim != qwen_hidden:
                self.extra_proj = nn.Linear(bridging_dim, qwen_hidden)
                print(f"[Info] bridging_dim={bridging_dim} != qwen_hidden={qwen_hidden}, 创建 extra_proj 做适配")
            else:
                self.extra_proj = None
        else:
            self.extra_proj = None

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None
    ):
        # 1) vision => bridging
        vision_embeds = self.vision_encoder(pixel_values)
        if self.extra_proj is not None:
            vision_embeds = self.extra_proj(vision_embeds)

        # 2) text => Qwen embedding
        text_embeds = self.qwen_model.get_input_embeddings()(input_ids)
        full_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

        if attention_mask is not None:
            bsz, t_seq = attention_mask.shape
            v_seq = vision_embeds.shape[1]
            vision_mask = attention_mask.new_ones((bsz, v_seq))
            full_mask = torch.cat([vision_mask, attention_mask], dim=1)
        else:
            full_mask = None

        outputs = self.qwen_model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            labels=labels
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens=128,
        **gen_kwargs
    ):
        vision_embeds = self.vision_encoder(pixel_values)
        if self.extra_proj is not None:
            vision_embeds = self.extra_proj(vision_embeds)

        text_embeds = self.qwen_model.get_input_embeddings()(prompt_ids)
        full_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

        bsz, _ = prompt_ids.shape
        v_seq = vision_embeds.shape[1]
        vision_mask = attention_mask.new_ones((bsz, v_seq))
        full_mask = torch.cat([vision_mask, attention_mask], dim=1)

        gen_out = self.qwen_model.generate(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            max_new_tokens=max_new_tokens,
            **gen_kwargs
        )
        return gen_out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # A) Load llava-phi-3-mini-hf
    llava_model_id = "/root/autodl-tmp/model/llava-phi-3-mini-hf"
    llava_full = LlavaForConditionalGeneration.from_pretrained(
        llava_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    llava_full.eval()

    # 构造 vision encoder
    llava_vision_encoder = LlavaPhiVisionEncoder(llava_full)
    # Freeze vision
    for p in llava_vision_encoder.parameters():
        p.requires_grad = False

    # B) Load QwenMath 2.5
    qwenmath_id = "/root/autodl-tmp/model/Qwenmath_2.5_1.5b"
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwenmath_id, use_fast=False)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        qwenmath_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    qwen_model.eval()

    # C) Combine
    multi_model = MultiModalQwen(llava_vision_encoder, qwen_model).to(device)

    # D) Prepare image & processor
    processor = AutoProcessor.from_pretrained(llava_model_id)
    # 修正警告：设置 patch_size 和 vision_feature_select_strategy
    # (具体数值/策略请查阅你所使用的LLaVa变体说明)
    processor.patch_size = 14
    processor.vision_feature_select_strategy = "patch"  # 或 "token"/"entire"/"cls"/"mean_pool" 等

    # 下载图片
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    # 调用 processor 时可传 text="" 避免 None
    image_inputs = processor(
        text="",
        images=raw_image,
        return_tensors='pt'
    )
    pixel_values = image_inputs["pixel_values"].to(device, dtype=torch.float16 if device == "cuda" else torch.float32)

    # E) Prepare text
    user_prompt = "描述一下这张图片中的场景，并回答：如果再加上3只狗，那么总共有多少只动物？"
    text_inputs = qwen_tokenizer(user_prompt, return_tensors='pt')
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    # F) Inference
    with torch.no_grad():
        gen_out = multi_model.generate(
            pixel_values=pixel_values,
            prompt_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )

    generated_text = qwen_tokenizer.decode(gen_out[0], skip_special_tokens=True)
    print("=== 模型生成结果 ===\n", generated_text)


if __name__ == "__main__":
    main()
