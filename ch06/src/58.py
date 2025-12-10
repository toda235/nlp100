from gensim.models import KeyedVectors
import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

load_dotenv()

model_path = os.getenv("GoogleNews_Vector")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# ファイル読み込み
df = pd.read_csv(
    "data/questions-words.txt",
    header=None,
    names=["1","2","3","4"],
    sep=r"\s+",
    comment=":"
)

# 国名リスト
countries = df["4"].unique()

# モデルに存在する単語だけ採用
vec = []
names = []

for c in countries:
    if c in model:
        vec.append(model[c])
        names.append(c)

X = np.array(vec)

# Ward 法による階層型クラスタリング
Z = linkage(X, method='ward')

# デンドログラム描画
plt.figure(figsize=(12, 6))
dendrogram(Z, labels=names, leaf_rotation=90)
plt.title("Clustering of Country Word Vectors")
plt.tight_layout()
plt.savefig("output/58.png", dpi=300)
plt.show()