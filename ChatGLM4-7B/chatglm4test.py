import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

# 设置模型下载缓存目录
cache_dir = "/root/autodl-tmp/model/GLM4-7Bchat"

# 下载并加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("ZhipuAI/glm-4-9b-chat", trust_remote_code=True, cache_dir=cache_dir)

query = "你好"

inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

inputs = inputs.to(device)

# 下载并加载模型
model = AutoModelForCausalLM.from_pretrained(
    "ZhipuAI/glm-4-9b-chat",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    cache_dir=cache_dir
).to(device).eval()

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
