#第7章のデータを使用した．（Google Newsデータセット）
from gensim.models import KeyedVectors
import numpy as np
import csv
import torch
from dotenv import load_dotenv
import os


load_dotenv()
model_path = os.getenv("GoogleNews_Vector")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

vocab_size = len(model.key_to_index)
emb_dim = model.vector_size

word_to_id = {"<PAD>":0}
id_to_word = {0:"<PAD>"}

embedding_matrix = np.zeros((vocab_size+1, emb_dim))

for word, gensim_id in model.key_to_index.items():
    word_id = gensim_id + 1
    word_to_id[word] = word_id
    id_to_word[word_id] = word
    embedding_matrix[word_id] = model[word]

def load_sst2(file_path, word_to_id):
    data = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row["sentence"]
            label = float(row["label"])

            words = text.split()
            input_ids = [word_to_id[w] for w in words if w in word_to_id]

            if len(input_ids) == 0:
                continue

            data.append({
                "text": text,
                "label": torch.tensor([label]),
                "input_ids": torch.tensor(input_ids)
            })
    return data

train_path = "ch07/data/SST-2/train.tsv"
dev_path= "ch07/data/SST-2/dev.tsv"

train_data = load_sst2(train_path, word_to_id)
dev_data = load_sst2(dev_path, word_to_id)

# print(f"train_data: {len(train_data)}")
# print(f"dev_data: {len(dev_data)}")
print(train_data[0])
