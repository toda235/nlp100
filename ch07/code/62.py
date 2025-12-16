import csv
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

text = []
label = []
train_data = []
dev_data = []

with open("data/SST-2/train.tsv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for i, row in enumerate(reader):
        text = row["sentence"]
        label = row["label"]
        
        words = text.split()
        cnt = dict(Counter(words))
        
        new_row = {
            'text' : text,
            'label' : label,
            'feature' : cnt
        }
        train_data.append(new_row)

with open("data/SST-2/dev.tsv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for i, row in enumerate(reader):
        text = row["sentence"]
        label = row["label"]
        
        words = text.split()
        cnt = dict(Counter(words))
        
        new_row = {
            'text' : text,
            'label' : label,
            'feature' : cnt
        }
        dev_data.append(new_row)
        
vec = DictVectorizer()
x_train = vec.fit_transform(j['feature'] for j in train_data)
y_train = [j['label'] for j in train_data]

x_dev = vec.transform(j['feature'] for j in dev_data)
y_dev = [j['label'] for j in dev_data]


lr = LogisticRegression(max_iter=1000)
lr.fit(x_train,y_train)

y_pred = lr.predict(x_dev)
acc = accuracy_score(y_dev, y_pred)
print(f"Accuracy: {acc}")