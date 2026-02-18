import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()

prompt = "The movie was full of"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,          
        do_sample=False,            # greedy decoding確率が最大
        return_dict_in_generate=True,
        output_scores=True
    )

# 生成されたトークンID
generated_ids = outputs.sequences[0]

# 入力長
input_length = inputs["input_ids"].shape[1]

print("=== Generated Text ===")
print(tokenizer.decode(generated_ids))
print("\n=== Token Likelihoods ===")

for i, score in enumerate(outputs.scores):
    probs = F.softmax(score, dim=-1)
    token_id = generated_ids[input_length + i]
    token_prob = probs[0, token_id].item()
    token_str = tokenizer.decode(token_id)

    print(f"Token: {token_str:>10} | Probability: {token_prob:.6f}")