from gensim.models import KeyedVectors
import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

load_dotenv()

model_path = os.getenv("GoogleNews_Vector")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

df = pd.read_csv(
    "data/questions-words.txt",
    header=None,
    names=["1","2","3","4"],
    sep=r"\s+",
    comment=":"
)

countries = df["4"].unique()

vec = []
names = []

for c in countries:
    if c in model:
        vec.append(model[c])
        names.append(c)

X = np.array(vec)

# t-SNE (2次元)
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
X_tsne = tsne.fit_transform(X)

# 可視化
plt.figure(figsize=(12, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)

for i, name in enumerate(names):
    plt.text(X_tsne[i, 0], X_tsne[i, 1], name, fontsize=8)

plt.title("t-SNE of Country Word Vectors")
plt.tight_layout()
plt.savefig("output/59.png")
plt.show()