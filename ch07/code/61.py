import csv
from collections import Counter

text = []
label = []

print("===== train =====")

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
        
        print(new_row)
        if i > 3:
            break
        
print("===== dev =====")

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
        
        print(new_row)
        if i > 3:
            break