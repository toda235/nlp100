#第6章のデータを使用した．（Google Newsデータセット）
from gensim.models import KeyedVectors
import numpy as np
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
    
print(embedding_matrix.shape)
print(f"|V|:{embedding_matrix.shape[0]}")
print(f"d_emb:{embedding_matrix.shape[1]}")

# print("vector_sample:", embedding_matrix[0][:5])
