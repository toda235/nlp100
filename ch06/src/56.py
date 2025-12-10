from gensim.models import KeyedVectors
import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

model_path = os.getenv("GoogleNews_Vector")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

df = pd.read_csv("data/combined.csv")
word1=df["Word 1"]
word2=df["Word 2"]
humun=df["Human (mean)"]

score=[]
for word1,word2 in zip(word1,word2):
    v1=model[word1]
    v2=model[word2]
    sim=cos(v1,v2)
    score.append(sim)
    
df["cos_sim"]=score
spearman_corr = df[["Human (mean)", "cos_sim"]].corr(method="spearman").iloc[0, 1]
print(spearman_corr)
# print(df["cos_sim"].corr(method="spearman"))