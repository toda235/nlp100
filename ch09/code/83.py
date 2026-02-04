import torch
from transformers import AutoTokenizer,AutoModel
from sklearn.metrics.pairwise import cosine_similarity

texts = [
    "The movie was full of fun.",
    "The movie was full of excitement",
    "The movie was full of crap.",
    "The movie was full of rubbish"
] 

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

inputs = tokenizer(
    texts,
    padding = True,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)
    
cls_embeddings = outputs.last_hidden_state[:, 0, :]
cls_np = cls_embeddings.cpu().numpy()
cos_sim = cosine_similarity(cls_np)

print(cos_sim)