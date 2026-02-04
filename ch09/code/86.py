import torch
from transformers import AutoTokenizer
import pandas as pd


def load_sst2(path):
    df = pd.read_table(path)
    texts = df["sentence"].tolist()
    labels = df["label"].tolist()
    
    return texts,labels
    
def tokenize_texts(texts, tokenizer):
    inputs = tokenizer(
        texts,
        padding = True,
        truncation=True,
        return_tensors = "pt"
    )
    return inputs

def main():
    model_name = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    
    train_texts, train_labels = load_sst2("../ch07/data/SST-2/train.tsv")
    dev_texts,dev_labels = load_sst2("../ch07/data/SST-2/dev.tsv")
    
    train_inputs = tokenize_texts(train_texts[:4],tokenizer)
    train_inputs_labels = torch.tensor(train_labels[:4])
    
    dev_inputs = tokenize_texts(dev_texts[:4],tokenizer)
    dev_inputs_labels = torch.tensor(dev_labels[:4])
    

    # 確認
    print("train:", len(train_texts))
    for key, value in train_inputs.items():
        print(key, value[0])
    print(train_inputs["input_ids"].shape)
    print(train_inputs["attention_mask"].shape)
    print(dev_inputs_labels.shape)
    print("===")
    print("dev:", len(dev_texts))
    for key, value in dev_inputs.items():
        print(key, value[0])
    print(dev_inputs["input_ids"].shape)
    print(dev_inputs["attention_mask"].shape)
    print(dev_inputs_labels.shape)

    
    train_labels = torch.tensor(train_labels)
    dev_labels   = torch.tensor(dev_labels)


if __name__ == "__main__":
    main()
