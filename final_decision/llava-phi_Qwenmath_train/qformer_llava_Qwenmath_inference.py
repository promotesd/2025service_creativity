from transformers import LlavaForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from qformer_llava_Qwenmath_model import LlavaPhiVisionEncoder, AdvancedMultiModalQwen
import torch
from PIL import Image
import requests

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load LLaVA-Phi-3-Mini-HF
    llava_model_path = "/root/autodl-tmp/model/llava-phi-3-mini-hf"
    llava_full = LlavaForConditionalGeneration.from_pretrained(
        llava_model_path,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32
    ).to(device).eval()

    # 2) Build Vision Encoder
    llava_vision_encoder = LlavaPhiVisionEncoder(llava_full)
    for p in llava_vision_encoder.parameters():
        p.requires_grad = False

    # 3) Load QwenMath
    qwenmath_path = "/root/autodl-tmp/model/Qwenmath_2.5_1.5b"
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwenmath_path, use_fast=False)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        qwenmath_path,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32
    ).to(device).eval()

    # 4) Build advanced multi-modal model
    multi_model = AdvancedMultiModalQwen(
        llava_vision_encoder=llava_vision_encoder,
        qwen_model=qwen_model,
        qformer_layers=4,
        qformer_heads=8,
        qformer_hidden_dim=1536,
        use_mrope=True
    ).to(device)

    # 5) Processor
    processor = AutoProcessor.from_pretrained(llava_model_path)
    processor.patch_size = 14
    processor.size = {"height": 336, "width": 336}
    processor.vision_feature_select_strategy = "default"

    # 6) Example input
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    image_inputs = processor(text="", images=raw_image, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device, dtype=torch.float16 if device=="cuda" else torch.float32)

    user_prompt = "请描述图像中的物体，并用公式表示数量关系。"
    text_inputs = qwen_tokenizer(user_prompt, return_tensors="pt")
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    # 7) Generate
    gen_out = multi_model.generate(
        pixel_values=pixel_values,
        prompt_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )
    out_ids = gen_out[0]
    print("=== Model Output ===")
    print(qwen_tokenizer.decode(out_ids, skip_special_tokens=True))

if __name__=="__main__":
    main()
