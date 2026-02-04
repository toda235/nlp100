import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

texts = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

model_dir = "../ch09/result/checkpoint-2106"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

model.eval()

label = {0: "negative", 1: "positive"}

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**encodings)
    logits = outputs.logits

preds = torch.argmax(logits, dim=-1).tolist()

for text, pred in zip(texts, preds):
    print(text)
    print(f'  label: {label[pred]}')
