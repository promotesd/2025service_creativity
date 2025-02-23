#!/usr/bin/env python
# coding: utf-8

"""
示例：使用更复杂的双向+多层 Q-Former结构来融合 LLaVA-Phi-3-Mini-HF 视觉特征和
      QwenMath_2.5_1.5b 语言特征，并可选结合 GeometryFormalizer 解析几何图。
官方推荐：patch_size=14, 图像分辨率 336×336, vision_feature_select_strategy="default"。
需要在多模态数据上微调，才能真正发挥高精度。
"""

import torch
import torch.nn as nn
from PIL import Image
import requests
from typing import Optional, Tuple

# pip install transformers accelerate safetensors sentencepiece
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer
)

# ===================================================
# 1) QFormerBlock: 基础双向 Cross-Attention 单元
# ===================================================

class QFormerAttentionBlock(nn.Module):
    """
    一个小型的Transformer block，实现:
    - text -> vision 的 cross attention
    - vision -> text 的 cross attention
    - MLP feed-forward, 残差、LayerNorm等
    简化写法，用PyTorch自带MultiheadAttention。
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # text->vision cross-attn
        self.cross_attn_t2v = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm_t2v_t = nn.LayerNorm(hidden_dim)
        self.norm_t2v_v = nn.LayerNorm(hidden_dim)

        # vision->text cross-attn
        self.cross_attn_v2t = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm_v2t_t = nn.LayerNorm(hidden_dim)
        self.norm_v2t_v = nn.LayerNorm(hidden_dim)

        # feed-forward for text
        self.mlp_text = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_text_ff = nn.LayerNorm(hidden_dim)

        # feed-forward for vision
        self.mlp_vision = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_vision_ff = nn.LayerNorm(hidden_dim)

    def forward(self, text_states: torch.Tensor, vision_states: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None,
                vision_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        text_states : [B, T, hidden_dim]
        vision_states: [B, V, hidden_dim]
        text_mask   : [B, T]  (可选)
        vision_mask : [B, V]
        return (updated_text, updated_vision)
        """

        # 1) text->vision cross-attn:  让 vision 作为 Key/Value, text 作为 Query
        #    text_states 被更新， vision 暂不变
        # batch_first=True => shape: (B, seq, dim)
        # MultiheadAttention: attn(Q,K,V)
        # Q = text, K=vision, V=vision
        residual_text = text_states
        text2, _ = self.cross_attn_t2v(
            query=text_states,
            key=vision_states,
            value=vision_states,
            key_padding_mask=~vision_mask if vision_mask is not None else None,
            need_weights=False
        )
        # 残差 + LN
        text2 = residual_text + text2
        text2 = self.norm_t2v_t(text2)

        # vision这边不更新，但可视需求加
        updated_vision = vision_states  # no update
        # 如果要做对 vision 的 LN:
        updated_vision = self.norm_t2v_v(updated_vision)

        # 2) vision->text cross-attn: 让 text 作为 Key/Value, vision 作为 Query
        #    vision_states 被更新
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

        # text 也可再更新(可选) => 例如 residual
        updated_text = text2
        updated_text = self.norm_v2t_t(updated_text)  # 仅 LN

        # 3) FFN for text
        #   text2 => residual
        text_ff_res = updated_text
        text_ff_out = self.mlp_text(updated_text)
        text_ff_out = text_ff_res + text_ff_out
        text_ff_out = self.norm_text_ff(text_ff_out)

        # 4) FFN for vision
        vision_ff_res = vision2
        vision_ff_out = self.mlp_vision(vision2)
        vision_ff_out = vision_ff_res + vision_ff_out
        vision_ff_out = self.norm_vision_ff(vision_ff_out)

        return text_ff_out, vision_ff_out


# ===================================================
# 2) QFormer: 多层堆叠 QFormerBlock
#    负责将 [text_embeds, vision_embeds] 做多轮双向 cross-attn 融合
# ===================================================

class QFormer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 num_layers: int,
                 # 也可加位置编码 etc
                 ):
        super().__init__()
        self.layers = nn.ModuleList([
            QFormerAttentionBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self,
                text_embeds: torch.Tensor,
                vision_embeds: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None,
                vision_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        text_embeds: [B, T, hidden_dim]
        vision_embeds: [B, V, hidden_dim]
        returns: (fused_text_embeds, fused_vision_embeds)
        """
        t_out, v_out = text_embeds, vision_embeds
        for layer in self.layers:
            t_out, v_out = layer(
                text_states=t_out,
                vision_states=v_out,
                text_mask=text_mask,
                vision_mask=vision_mask
            )
        return t_out, v_out


# ===================================================
# 3) 几何Formalizer: 与之前相同
# ===================================================
class GeometryFormalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, vision_embeds: torch.Tensor, question_text: str) -> str:
        if "triangle" in question_text.lower():
            geometry_formal_string = (
                "Detected a triangle in the figure. Formal constraints: angles sum to 180, sides relation unknown.\n"
            )
        elif "function" in question_text.lower():
            geometry_formal_string = (
                "Detected a function curve. Possibly a polynomial or sin/cos. Additional formal info: f(x)=...\n"
            )
        else:
            geometry_formal_string = ""
        return geometry_formal_string


# ===================================================
# 4) LlavaPhiVisionEncoder: 只做图像->ViT->初步embedding
#    不再做 linear_2 bridging。让 QFormer 来负责融合
# ===================================================
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

        # 注：不再使用 llava_model.model.visual_projection 直接投影
        # 因为我们要让 QFormer 来做更复杂的 cross-attn
        # 你若想保留 bridging，也可self.bridging = ...
        self.hidden_dim = 1024  # LLaVA-Phi-3-mini: CLIP-ViT-L/14 => 1024

    def forward(self, images: torch.Tensor):
        outputs = self.vision_tower(images)  # CLIPVisionModelOutput
        vision_feats = outputs.last_hidden_state  # [batch, seq_len, 1024]
        return vision_feats


# ===================================================
# 5) AdvancedMultiModalQwen: 整体封装
#    - vision => LlavaPhiVisionEncoder => vision_feats
#    - text => qwen embedding => text_embeds
#    - 交给 QFormer 做多层双向 cross-attn => (text_fused, vision_fused)
#    - 如果 geometry_formalizer, 在 text_fused 基础上可追加
#    - 最后只把 text_fused 喂给 Qwen (或把 vision_fused 拼上)
# ===================================================
class AdvancedMultiModalQwen(nn.Module):
    def __init__(
        self,
        llava_vision_encoder: nn.Module,
        qwen_model: nn.Module,
        geometry_formalizer: nn.Module = None,
        qformer_layers: int = 4,
        qformer_heads: int = 8,
        qformer_hidden_dim: int = 1536
    ):
        """
        qformer_hidden_dim 要与 Qwen hidden_size 对齐(=1536),
        但图像是 1024 => 需要单独的proj?
        """
        super().__init__()
        self.vision_encoder = llava_vision_encoder
        self.qwen_model = qwen_model
        self.geometry_formalizer = geometry_formalizer

        # 视觉输出是 [batch, v_seq, 1024], Qwen hidden=1536 => 先线性把 1024 -> 1536
        self.vis_proj = nn.Linear(1024, qformer_hidden_dim)

        # QFormer (多层双向 cross-attn)
        self.qformer = QFormer(
            hidden_dim=qformer_hidden_dim,
            num_heads=qformer_heads,
            num_layers=qformer_layers
        )

        # QWen tokenizer embedding dim => 1536
        # -> no extra needed if text is already 1536

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        question_text: str = None,
    ):
        """
        1) Vision => raw feats => proj => v_emb
        2) text => qwen embedding => t_emb
        3) QFormer(t_emb, v_emb) => (t_fused, v_fused)
        4) geometry_formalizer => ...
        5) pass t_fused to qwen language model
        """
        # A) vision feats
        vision_feats = self.vision_encoder(pixel_values)  # [B, v_seq, 1024]
        vision_embeds = self.vis_proj(vision_feats)       # [B, v_seq, 1536]

        # B) text => qwen embedding
        text_embeds = self.qwen_model.get_input_embeddings()(input_ids)  # [B, t_seq, 1536]

        # C) QFormer
        #    这里为了简化mask, 视 vision_mask/text_mask 全为1
        #    你可根据 attention_mask==0 => mask
        if attention_mask is not None:
            text_mask = attention_mask.bool()  # [B, t_seq], True=not masked
        else:
            text_mask = None
        # vision_mask 全1
        bsz, v_seq, _ = vision_embeds.shape
        vision_mask = vision_embeds.new_ones((bsz, v_seq), dtype=torch.bool)

        t_fused, v_fused = self.qformer(
            text_embeds, vision_embeds,
            text_mask=text_mask, vision_mask=vision_mask
        )
        # t_fused: [B, t_seq, 1536], v_fused: [B, v_seq, 1536]

        # D) geometry_formalizer
        #   如果question_text含“triangle”/“function”，就生成 add_text
        #   这里仅演示 => 并未真正把 add_text encode 进 t_fused
        #   你可另写 “二次 tokenize => QFormer” 的逻辑
        formal_text = ""
        if self.geometry_formalizer and question_text:
            # note: v_fused or vision_embeds for geometry parse
            formal_text = self.geometry_formalizer(v_fused, question_text)

        # E) Qwen decoder: 先替换embedding?
        #   我们可以只用 t_fused 作为 “inputs_embeds”，跟原 attention_mask 对齐
        outputs = self.qwen_model(
            inputs_embeds=t_fused,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs, formal_text

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        question_text: str,
        max_new_tokens=128,
        **gen_kwargs
    ):
        """
        推理流程:
        1) vision => encoder => proj
        2) text => qwen embedding
        3) QFormer => (t_fused, v_fused)
        4) geometry_formalizer => ...
        5) qwen_model.generate(inputs_embeds=t_fused)
        """
        # A) vision
        vision_feats = self.vision_encoder(pixel_values)  # [B, v_seq, 1024]
        vision_embeds = self.vis_proj(vision_feats)       # [B, v_seq, 1536]

        # B) text
        text_embeds = self.qwen_model.get_input_embeddings()(prompt_ids)

        bsz, t_seq, _ = text_embeds.shape
        v_seq = vision_embeds.shape[1]

        # attention mask
        if attention_mask is not None:
            text_mask = attention_mask.bool()  # True=keep
        else:
            text_mask = None
        vision_mask = vision_embeds.new_ones((bsz, v_seq), dtype=torch.bool)

        # C) QFormer
        t_fused, v_fused = self.qformer(
            text_embeds, vision_embeds,
            text_mask=text_mask, vision_mask=vision_mask
        )

        # D) geometry
        formal_text = ""
        if self.geometry_formalizer and question_text:
            formal_text = self.geometry_formalizer(v_fused, question_text)

        # E) 生成
        #   huggingface generate 不直接支持 halfway inputs_embeds -> continuing
        #   这里我们一次性把 t_fused 作为 prefix，再继续 decode
        gen_out = self.qwen_model.generate(
            inputs_embeds=t_fused,
            attention_mask=attention_mask,  # 仍用 text的mask
            max_new_tokens=max_new_tokens,
            **gen_kwargs
        )
        return gen_out, formal_text


# ===================================================
# 6) main(): 实例化 & 测试
# ===================================================
# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # A) 加载 llava-phi-3-mini-hf
#     llava_model_id = "/root/autodl-tmp/model/llava-phi-3-mini-hf"
#     llava_full = LlavaForConditionalGeneration.from_pretrained(
#         llava_model_id,
#         torch_dtype=torch.float16 if device == "cuda" else torch.float32,
#         low_cpu_mem_usage=True,
#     ).to(device)
#     llava_full.eval()

#     # 构造 vision encoder
#     llava_vision_encoder = LlavaPhiVisionEncoder(llava_full)
#     for p in llava_vision_encoder.parameters():
#         p.requires_grad = False

#     # B) QwenMath_2.5
#     qwenmath_id = "/root/autodl-tmp/model/Qwenmath_2.5_1.5b"
#     qwen_tokenizer = AutoTokenizer.from_pretrained(qwenmath_id, use_fast=False)
#     qwen_model = AutoModelForCausalLM.from_pretrained(
#         qwenmath_id,
#         torch_dtype=torch.float16 if device == "cuda" else torch.float32
#     ).to(device)
#     qwen_model.eval()

#     # C) 组合
#     geometry_formalizer = GeometryFormalizer()
#     multi_model = AdvancedMultiModalQwen(
#         llava_vision_encoder, qwen_model,
#         geometry_formalizer=geometry_formalizer,
#         qformer_layers=4,    # 多层
#         qformer_heads=8,     # multi-head
#         qformer_hidden_dim=1536  # Qwen hidden
#     ).to(device)

#     # D) processor: patch=14, size=336, default => etc
#     processor = AutoProcessor.from_pretrained(llava_model_id)
#     processor.patch_size = 14
#     processor.size = {"height": 336, "width": 336}
#     processor.vision_feature_select_strategy = "default"

#     # E) 测试图像
#     image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
#     image_inputs = processor(text="", images=raw_image, return_tensors='pt')
#     pixel_values = image_inputs["pixel_values"].to(device, dtype=torch.float16 if device == "cuda" else torch.float32)

#     # F) 测试文本
#     user_prompt = "在这张图片里看到三角形吗？若存在，请给出几何描述和计算步骤。"
#     text_inputs = qwen_tokenizer(user_prompt, return_tensors='pt')
#     input_ids = text_inputs["input_ids"].to(device)
#     attention_mask = text_inputs["attention_mask"].to(device)

#     # G) 生成
#     with torch.no_grad():
#         gen_out, formal_text = multi_model.generate(
#             pixel_values=pixel_values,
#             prompt_ids=input_ids,
#             attention_mask=attention_mask,
#             question_text=user_prompt,  # 传给 geometry_formalizer
#             max_new_tokens=100,
#             do_sample=True,
#             top_p=0.9,
#             temperature=0.8
#         )

#     # decode
#     generated_ids = gen_out[0]
#     generated_text = qwen_tokenizer.decode(generated_ids, skip_special_tokens=True)

#     print("=== 几何Formalizer 额外文本 ===")
#     print(formal_text)
#     print("=== 模型生成结果 ===\n", generated_text)


# # ===================================================
# # 7) 提示: 多阶段训练 & 数据增强
# # ===================================================
# """
# 要让本Q-Former真正发挥数学/几何推理威力，需要:
# 1) 大规模多模态训练, 包含大量函数图/几何图
# 2) 多阶段微调, 与QwenMath结合(文本数学) => 最后在多模态数学数据集(如SynthGeo,MathV360K等)上做SFT
# 3) 若有外部工具(如符号计算器), 可在Qwen2.5-Math中注入ToolCall, 进一步提升复杂运算能力
# 4) RLHF等过程监督, 优化多模态推理质量

# 这样, 就能实现在图像解析 + 几何Formalizer + Q-Former 多层交互 + QwenMath推理的强大多模态数学解题.
# """

# if __name__ == "__main__":
#     main()
