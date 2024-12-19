import torch
from transformers import pipeline
from modelscope import snapshot_download

# 下载模型到指定路径
model_dir = snapshot_download('LLM-Research/Llama-3.2-3B-Instruct', local_dir='/root/autodl-tmp/model/llama3.2-3B')

# 创建文本生成管道
pipe = pipeline(
    "text-generation",
    model=model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 输入消息
messages = [
    {"role": "system", "content": "You are a math teacher!"},
    {"role": "user", "content": "Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$"},
]

# 生成文本
outputs = pipe(
    messages,
    max_new_tokens=256,
)

# 打印生成的文本
print(outputs[0]["generated_text"][-1])
