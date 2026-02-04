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
    
# 最終層の隠れ状態
last_hidden_state = outputs.last_hidden_state
attention_mask = inputs["attention_mask"]

# 次元を合わせる
mask = attention_mask.unsqueeze(-1) 

# inputsでの調節したPADを除いて和を取る
masked_sum = (last_hidden_state * mask).sum(dim=1)
token_counts = mask.sum(dim=1)
mean_embeddings = masked_sum / token_counts

mean_np = mean_embeddings.cpu().numpy()
cos_sim = cosine_similarity(mean_np)

print(cos_sim)