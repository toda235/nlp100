import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

text = "The movie was full of"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

# 最後の位置（次トークン予測用）
next_token_logits = logits[:, -1, :]

probs = F.softmax(next_token_logits, dim=-1)

top_probs, top_ids = torch.topk(probs, 10)

for prob, token_id in zip(top_probs[0], top_ids[0]):
    word = tokenizer.decode(token_id)
    print(f"{word.strip()}: {prob.item():.4f}")