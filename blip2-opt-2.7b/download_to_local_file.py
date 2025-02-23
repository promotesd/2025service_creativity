import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# 强制指定 cache_dir = '/root/autodl-tmp/model/blip2-opt-2.7b'
processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    cache_dir="/root/autodl-tmp/model/blip2-opt-2.7b",
    force_download=True
)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    cache_dir="/root/autodl-tmp/model/blip2-opt-2.7b",
    device_map="auto",
    force_download=True
)

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

question = "how many dogs are in the picture?"
inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True).strip())
