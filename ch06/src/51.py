from gensim.models import KeyedVectors
import numpy as np
from dotenv import load_dotenv
import os

def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

load_dotenv()

model_path = os.getenv("GoogleNews_Vector")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

v1 = model["United_States"]
v2 = model["U.S."]

print(cos(v1,v2))