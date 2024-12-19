# 效果很差,

# To find the value of $x$ that satisfies the equation $4x+5 = 6x+7$, we can use the method of substitution.

# First, let's isolate the variable $x$ on one side of the equation:

# $4x + 5 - 6x - 7 = 0$

# $4x - 6x - 7 = -6x + 5$

# $x(4 - 6) - x(7) = x(5)$

# $x(4/6) - x(7/6) = x(5/6)$

# $x = \frac{5}{6}$

# So, the value of $x$ that satisfies the equation is $x = \frac{5}{6}$.
# Generation time: 411.5283 seconds

import time
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = '/root/autodl-tmp/model/Xwin-LM-7B'
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)

# Prompt
prompt = "You are a math teacher" \
         "USER: Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$? " \
         "ASSISTANT:"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# Start time
start_time = time.time()

# Generate output
samples = model.generate(**inputs, max_new_tokens=4096, temperature=0.7)
output = tokenizer.decode(samples[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# End time
end_time = time.time()

# Print output
print(output)

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Generation time: {elapsed_time:.4f} seconds")
