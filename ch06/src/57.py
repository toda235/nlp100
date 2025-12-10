from gensim.models import KeyedVectors
import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

load_dotenv()

model_path = os.getenv("GoogleNews_Vector")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

df = pd.read_csv(
    "data/questions-words.txt",
    header=None,
    names=["1","2","3","4"],
    delim_whitespace=True,
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

# K-means
kmeans = KMeans(n_clusters=5, random_state=0)
labels = kmeans.fit_predict(X)

# PCA
X_pca = PCA(n_components=2).fit_transform(X)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.7)

for i, name in enumerate(names):
    plt.text(X_pca[i, 0], X_pca[i, 1], name, fontsize=9, alpha=0.6)

plt.title("KMeans Clustering of Countries (Word2Vec)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
from gensim.models import KeyedVectors
import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

load_dotenv()

model_path = os.getenv("GoogleNews_Vector")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# df = pd.read_csv(
#     "data/questions-words.txt",
#     header=None,
#     names=["1","2","3","4"],
#     delim_whitespace=True,
#     comment=":"
# )

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

# K-means
kmeans = KMeans(n_clusters=5, random_state=0)
labels = kmeans.fit_predict(X)

# PCA
X_pca = PCA(n_components=2).fit_transform(X)

# plot
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.7)

for i, name in enumerate(names):
    plt.text(X_pca[i, 0], X_pca[i, 1], name, fontsize=9, alpha=0.6)

plt.title("KMeans Clustering of Countries")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.savefig("output/57.png")
plt.show()
