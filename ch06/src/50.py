from gensim.models import KeyedVectors
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

model_path = os.getenv("GoogleNews_Vector")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)


print(model["United_States"])