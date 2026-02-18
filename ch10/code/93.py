from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import math

model_name = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval() #評価

def perplexity(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["input_ids"]
        )

    loss = outputs.loss
    return math.exp(loss.item())
    

sentences = [
    "The movie was full of surprises",
    "The movies were full of surprises",
    "The movie were full of surprises",
    "The movies was full of surprises"
]

for i in sentences:
    predictability = perplexity(i)
    print(f"sentence:{i}")
    print(f"perplexity:{predictability:.5f}\n")

