#!/usr/bin/env python
# coding: utf-8

"""
qformer_llava_Qwenmath_model.py

更复杂的多模态桥接示例：
1) LLaVA-Phi-3-Mini-HF 提供视觉特征 (hidden=1024).
2) 可配置 VisionAggregator 将视觉特征聚合成多个"视觉 Token"，支持更大容量(qformer_hidden_dim可达2048)。
3) 双向+多层 Q-Former (可带M-RoPE) 与文本进行多轮交互，避免信息瓶颈。
4) 显式对齐 dtype，防止 half/float 冲突。
5) 适合在多模态数学场景下，通过更多视觉 Token & 更大桥接维度提升表达复杂图形的能力。

注：脚本无几何Formalizer；若需局部监督/关键点标注，可在 Aggregator 或 QFormer 中添加额外分支损失。
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import LlavaForConditionalGeneration, AutoModelForCausalLM

# ===================================================
# 0) M-RoPE: 多模态旋转位置嵌入
# ===================================================
class MultiModalRoPE(nn.Module):
    """
    多模态旋转位置编码 (M-RoPE).
    若需2D RoPE, 可基于(h,w)扩展; 此处仅1D, 对文本&视觉序列均可用.
    """

    def __init__(self, hidden_dim: int, max_position: int = 4096):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_position = max_position

        # 通常RoPE只对 hidden_dim 一半做 sin-cos
        self.half_dim = hidden_dim // 2

        # 构建 sin-cos table
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        position_ids = torch.arange(0, max_position).float().unsqueeze(1)
        freqs = position_ids * inv_freq.unsqueeze(0)
        self.register_buffer("sin_table", torch.sin(freqs), persistent=False)
        self.register_buffer("cos_table", torch.cos(freqs), persistent=False)

    def forward(self, x: torch.Tensor, seq_offset: int = 0) -> torch.Tensor:
        """
        x: [B, S, hidden_dim].
        seq_offset: 若增量offset; 示例中不用.
        """
        bsz, seq_len, dim = x.shape
        if dim != self.hidden_dim:
            raise ValueError(f"RoPE dimension mismatch: x={dim} vs hidden={self.hidden_dim}")

        sin_table = self.sin_table.to(x.device, dtype=x.dtype)
        cos_table = self.cos_table.to(x.device, dtype=x.dtype)

        # 将 x 前 half_dim 与后 half_dim 分开
        x1 = x[:, :, :self.half_dim]  # RoPE部分
        x2 = x[:, :, self.half_dim:]  # 不变部分

        sin_pos = sin_table[seq_offset: seq_offset+seq_len, :]
        cos_pos = cos_table[seq_offset: seq_offset+seq_len, :]

        half_half = self.half_dim // 2
        x1_reshaped = x1.view(bsz, seq_len, half_half, 2)
        x1_even = x1_reshaped[..., 0]
        x1_odd  = x1_reshaped[..., 1]

        sin_pos = sin_pos[:, :half_half].unsqueeze(0).expand(bsz, seq_len, half_half)
        cos_pos = cos_pos[:, :half_half].unsqueeze(0).expand(bsz, seq_len, half_half)

        rope_even = x1_even * cos_pos - x1_odd * sin_pos
        rope_odd  = x1_odd  * cos_pos + x1_even * sin_pos
        rope_x1 = torch.stack([rope_even, rope_odd], dim=-1).reshape(bsz, seq_len, self.half_dim)

        out = torch.cat([rope_x1, x2], dim=-1)
        return out


# ===================================================
# 1) VisionAggregator: 将视觉特征 -> 多聚合Token
# ===================================================
class VisionAggregator(nn.Module):
    """
    将视觉侧 [B, seq, in_dim] 转成 [B, aggregator_token_count, qformer_hidden_dim] (可大维度).
    实现: aggregator query 做 cross-attn to vision.
    若 aggregator_token_count=0 => 直接返回原 [B, seq, in_dim].
    """

    def __init__(
        self,
        in_dim: int = 1024,
        out_dim: int = 2048,
        aggregator_token_count: int = 8,
        num_heads: int = 8
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregator_token_count = aggregator_token_count
        self.num_heads = num_heads

        if aggregator_token_count > 0:
            # aggregator learnable query tokens
            self.query_tokens = nn.Parameter(torch.randn(aggregator_token_count, out_dim))

            # cross-attn aggregator
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=out_dim,
                num_heads=num_heads,
                batch_first=True
            )
            # project vision to out_dim
            self.vis_proj = nn.Linear(in_dim, out_dim)
            self.norm_vis = nn.LayerNorm(out_dim)
            self.norm_q   = nn.LayerNorm(out_dim)
        else:
            self.query_tokens = None

    def forward(self, vision_feats: torch.Tensor) -> torch.Tensor:
        """
        vision_feats: [B, seq, in_dim]
        return aggregator_out: [B, aggregator_token_count, out_dim] or [B, seq, in_dim] if aggregator=0
        """
        if self.aggregator_token_count <= 0:
            return vision_feats

        bsz, seq_len, in_dim = vision_feats.shape
        if in_dim != self.in_dim:
            raise ValueError(f"Aggregator in_dim mismatch: {in_dim} vs {self.in_dim}")

        # dtype alignment
        vision_feats = vision_feats.to(self.vis_proj.weight.dtype)

        # 1) project to out_dim
        vision_proj = self.vis_proj(vision_feats)
        vision_proj = self.norm_vis(vision_proj)  # [B, seq, out_dim]

        # 2) aggregator queries
        query = self.query_tokens.unsqueeze(0).expand(bsz, self.aggregator_token_count, self.out_dim)
        query = self.norm_q(query)

        # 3) cross-attn aggregator
        aggregator_out, _ = self.cross_attn(
            query=query,
            key=vision_proj,
            value=vision_proj,
            need_weights=False
        )
        # aggregator_out: [B, aggregator_token_count, out_dim]
        return aggregator_out


# ===================================================
# 2) QFormerBlock + QFormer (双向CrossAttn + M-RoPE)
# ===================================================
class QFormerAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, use_mrope: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_mrope = use_mrope

        self.cross_attn_t2v = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm_t2v_t = nn.LayerNorm(hidden_dim)
        self.norm_t2v_v = nn.LayerNorm(hidden_dim)

        self.cross_attn_v2t = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm_v2t_t = nn.LayerNorm(hidden_dim)
        self.norm_v2t_v = nn.LayerNorm(hidden_dim)

        self.mlp_text = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_text_ff = nn.LayerNorm(hidden_dim)

        self.mlp_vision = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_vision_ff = nn.LayerNorm(hidden_dim)

        if self.use_mrope:
            self.text_rope = MultiModalRoPE(hidden_dim)
            self.vision_rope = MultiModalRoPE(hidden_dim)
        else:
            self.text_rope = None
            self.vision_rope = None

    def forward(
        self,
        text_states: torch.Tensor,
        vision_states: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.use_mrope:
            text_states = self.text_rope(text_states)
            vision_states = self.vision_rope(vision_states)

        # text->vision
        residual_text = text_states
        text2, _ = self.cross_attn_t2v(
            query=text_states,
            key=vision_states,
            value=vision_states,
            key_padding_mask=~vision_mask if vision_mask is not None else None,
            need_weights=False
        )
        text2 = residual_text + text2
        text2 = self.norm_t2v_t(text2)

        updated_vision = vision_states
        updated_vision = self.norm_t2v_v(updated_vision)

        # vision->text
        residual_vision = updated_vision
        vision2, _ = self.cross_attn_v2t(
            query=updated_vision,
            key=text2,
            value=text2,
            key_padding_mask=~text_mask if text_mask is not None else None,
            need_weights=False
        )
        vision2 = residual_vision + vision2
        vision2 = self.norm_v2t_v(vision2)

        updated_text = text2
        updated_text = self.norm_v2t_t(updated_text)

        # FFN (text)
        text_ff_res = updated_text
        text_ff_out = self.mlp_text(updated_text)
        text_ff_out = text_ff_res + text_ff_out
        text_ff_out = self.norm_text_ff(text_ff_out)

        # FFN (vision)
        vision_ff_res = vision2
        vision_ff_out = self.mlp_vision(vision2)
        vision_ff_out = vision_ff_res + vision_ff_out
        vision_ff_out = self.norm_vision_ff(vision_ff_out)

        return text_ff_out, vision_ff_out


class QFormer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        use_mrope: bool = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            QFormerAttentionBlock(hidden_dim, num_heads, use_mrope=use_mrope)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        text_embeds: torch.Tensor,
        vision_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        t_out, v_out = text_embeds, vision_embeds
        for layer in self.layers:
            t_out, v_out = layer(t_out, v_out, text_mask=text_mask, vision_mask=vision_mask)
        return t_out, v_out


# ===================================================
# 3) LlavaPhiVisionEncoder: LLaVA-Phi => [B, seq, 1024]
# ===================================================
class LlavaPhiVisionEncoder(nn.Module):
    def __init__(self, llava_model: LlavaForConditionalGeneration):
        super().__init__()
        vision_tower = getattr(llava_model, "vision_tower", None)
        if vision_tower is None and hasattr(llava_model, "model"):
            vision_tower = getattr(llava_model.model, "vision_tower", None)
        if vision_tower is None:
            raise AttributeError("未能在 llava_model 中找到 vision_tower")

        if isinstance(vision_tower, (list, nn.ModuleList)):
            if len(vision_tower) == 0:
                raise ValueError("llava_model.vision_tower 是空")
            self.vision_tower = vision_tower[0]
        else:
            self.vision_tower = vision_tower

        self.hidden_dim = 1024

    def forward(self, images: torch.Tensor):
        outputs = self.vision_tower(images)
        vision_feats = outputs.last_hidden_state  # [B, seq, 1024]
        return vision_feats


# ===================================================
# 4) AdvancedMultiModalQwen: 整体桥接+QFormer => Qwen
# ===================================================
class AdvancedMultiModalQwen(nn.Module):
    """
    1) vision_encoder => [B, seq, 1024] patch特征
    2) aggregator => [B, aggregator_token_count, qformer_hidden_dim] or [B, seq, in_dim]
    3) Q-Former => (t_fused, v_fused)
    4) Qwen decode/generate

    => 通过 aggregator_token_count>0, qformer_hidden_dim>1024, 让模型更有容量
       并用M-RoPE+多层QFormer进行深度交互.
    """

    def __init__(
        self,
        llava_vision_encoder: LlavaPhiVisionEncoder,
        qwen_model: AutoModelForCausalLM,
        aggregator_token_count: int = 8,
        in_dim: int = 1024,              # LLaVA-Phi输出维度
        qformer_hidden_dim: int = 2048,  # aggregator & Q-Former hidden
        aggregator_heads: int = 8,
        qformer_layers: int = 4,
        qformer_heads: int = 8,
        use_mrope: bool = True
    ):
        super().__init__()
        self.vision_encoder = llava_vision_encoder
        self.qwen_model = qwen_model

        # VisionAggregator => [B, aggregator_token_count, qformer_hidden_dim]
        # or skip if aggregator_token_count=0
        self.aggregator = VisionAggregator(
            in_dim=in_dim,
            out_dim=qformer_hidden_dim,
            aggregator_token_count=aggregator_token_count,
            num_heads=aggregator_heads
        )

        # Q-Former
        self.qformer = QFormer(
            hidden_dim=qformer_hidden_dim,
            num_heads=qformer_heads,
            num_layers=qformer_layers,
            use_mrope=use_mrope
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None
    ):
        """
        训练/forward:
        1) vision encoder => aggregator => bridging feats
        2) text => embedding => bridging dtype
        3) Q-Former => (t_fused, v_fused)
        4) Qwen => outputs
        """
        # A) vision
        vision_feats = self.vision_encoder(pixel_values)
        # aggregator dtype
        aggregator_wt = self.aggregator.vis_proj.weight if self.aggregator.query_tokens is not None else None
        aggregator_dtype = aggregator_wt.dtype if aggregator_wt is not None else vision_feats.dtype
        vision_feats = vision_feats.to(aggregator_dtype)

        bridging_feats = self.aggregator(vision_feats)

        # B) text => Qwen embedding => bridging dtype
        text_embeds = self.qwen_model.get_input_embeddings()(input_ids)
        bridging_dtype = bridging_feats.dtype
        text_embeds = text_embeds.to(bridging_dtype)

        # C) masks
        if attention_mask is not None:
            text_mask = attention_mask.bool()
        else:
            text_mask = None
        bsz, v_seq, hidden_dim = bridging_feats.shape
        vision_mask = bridging_feats.new_ones((bsz, v_seq), dtype=torch.bool)

        # D) Q-Former
        t_fused, v_fused = self.qformer(
            text_embeds, bridging_feats,
            text_mask=text_mask,
            vision_mask=vision_mask
        )

        # E) Qwen decode
        qwen_dtype = self.qwen_model.get_input_embeddings().weight.dtype
        t_fused = t_fused.to(qwen_dtype)
        outputs = self.qwen_model(
            inputs_embeds=t_fused,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
        **gen_kwargs
    ):
        """
        推理:
        1) vision => aggregator => bridging feats
        2) text => embedding => bridging dtype
        3) Q-Former => (t_fused, v_fused)
        4) Qwen.generate(inputs_embeds=t_fused)
        """
        # A) vision
        vision_feats = self.vision_encoder(pixel_values)
        aggregator_wt = self.aggregator.vis_proj.weight if self.aggregator.query_tokens is not None else None
        aggregator_dtype = aggregator_wt.dtype if aggregator_wt is not None else vision_feats.dtype
        vision_feats = vision_feats.to(aggregator_dtype)

        bridging_feats = self.aggregator(vision_feats)

        # B) text
        text_embeds = self.qwen_model.get_input_embeddings()(prompt_ids)
        text_embeds = text_embeds.to(bridging_feats.dtype)

        bsz, v_seq, hidden_dim = bridging_feats.shape
        if attention_mask is not None:
            text_mask = attention_mask.bool()
        else:
            text_mask = None
        vision_mask = bridging_feats.new_ones((bsz, v_seq), dtype=torch.bool)

        # C) Q-Former
        t_fused, v_fused = self.qformer(
            text_embeds, bridging_feats,
            text_mask=text_mask,
            vision_mask=vision_mask
        )

        # D) Qwen generate
        qwen_dtype = self.qwen_model.get_input_embeddings().weight.dtype
        t_fused = t_fused.to(qwen_dtype)

        gen_out = self.qwen_model.generate(
            inputs_embeds=t_fused,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **gen_kwargs
        )
        return gen_out
