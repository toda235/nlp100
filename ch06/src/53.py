from gensim.models import KeyedVectors
import numpy as np
from dotenv import load_dotenv
import os


load_dotenv()

model_path = os.getenv("GoogleNews_Vector")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

similar_words = model.most_similar(positive=["Spain","Athens"], negative=["Madrid"], topn=10) 

print(similar_words)
