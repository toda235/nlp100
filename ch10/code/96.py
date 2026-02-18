import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_sst2(path):
    with open(path, encoding="utf-8") as f:
        data = f.read().rstrip().split("\n")
        data = [item.split("\t") for item in data[1:]]

    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]

    return sentences, labels

sentences, labels = load_sst2("../ch07/data/SST-2/dev.tsv")
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

model.eval()

def predict_sentiment(text):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a sentiment classifier. If positive output 1. If negative output 0. Output only one digit."
            )
        },
        {
            "role": "user",
            "content": text
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    if len(generated_text) > 0 and generated_text[0] in {"0", "1"}:
        return int(generated_text[0])
    else:
        return -1

correct = 0
total = len(sentences)
skipped = 0

for text, gold in zip(sentences, labels):
    pred = predict_sentiment(text)

    if pred == -1:
        skipped += 1
        continue

    if pred == gold:
        correct += 1

accuracy = correct / (total - skipped)

print(f"Accuracy : {accuracy:.4f}")
