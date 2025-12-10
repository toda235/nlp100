from gensim.models import KeyedVectors
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

model_path = os.getenv("GoogleNews_Vector")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)


data = []
current_section = None

with open("data/questions-words.txt", "r") as f:
    for line in f:
        line = line.strip()

        if line.startswith(":"):
            current_section = line[2:].strip()   # ":" と空白を除いた名前
            continue

        # 単語4つの行の場合のみ処理
        parts = line.split()
        if len(parts) == 4:
            w1, w2, w3, w4 = parts
            data.append([current_section, w1, w2, w3, w4])

df = pd.DataFrame(data, columns=["section", "w1", "w2", "w3", "w4"])

capital_df = df[df["section"] == "capital-common-countries"]

results = []

for _, row in capital_df.iterrows():
    w1, w2, w3, gt = row["w1"], row["w2"], row["w3"], row["w4"]

    if any(w not in model for w in [w1, w2, w3]):
        continue

    vec = model[w2] - model[w1] + model[w3]
    pred, sim = model.similar_by_vector(vec, topn=1)[0]

    results.append([w1, w2, w3, gt, pred, sim])


result_df = pd.DataFrame(
    results,
    columns=["1", "2", "3", "cor", "pred", "sim"]
)

print(result_df)



