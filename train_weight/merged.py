import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# 1) 原始Qwen基底模型路径
base_model_path = "/root/autodl-tmp/model/Qwenmath_2.5_1.5b"

# 2) LoRA微调后保存的文件夹(你训练完 "trainer.save_model()" 那个)
#    例如 /root/autodl-tmp/code/2025service_creativity/train_weight/numina_math_co_t/train_lora_20250223_164338
lora_model_path = "/root/autodl-tmp/code/2025service_creativity/train_weight/numina_math_co_t/train_lora_20250223_164338"

# 3) 加载原始基底模型
print("[INFO] Loading base model ...")
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)

# 4) 用 PeftModel 包装这个 base_model，并加载 LoRA 适配器
print("[INFO] Loading LoRA adapter from", lora_model_path)
peft_model = PeftModel.from_pretrained(base_model, lora_model_path)

# 5) 将 LoRA 权重合并进 base_model，并把 LoRA 卸载
print("[INFO] Merging LoRA weights into base model ...")
merged_model = peft_model.merge_and_unload()
# 这样 merged_model 现在是带有合并后权重的纯 transformers 模型

# 6) 另存合并结果
merged_model_save_path = "/root/autodl-tmp/model/Qwenmath_2.5_1.5b_merged_lora"
print("[INFO] Saving merged model =>", merged_model_save_path)
merged_model.save_pretrained(merged_model_save_path)

# 还可把 tokenizer 一并拷贝
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
tokenizer.save_pretrained(merged_model_save_path)

print("[INFO] Done. The model in", merged_model_save_path, "is now a single model with LoRA merged.")
