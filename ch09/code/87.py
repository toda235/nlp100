import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def load_sst2(path):
    df = pd.read_table(path)
    texts = df["sentence"].tolist()
    labels = df["label"].tolist()
    return texts, labels


# Trainerにデータを渡すために、Datasetクラスを作成
class SST2Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # 辞書形式 {input_ids: ..., attention_mask: ..., labels: ...} を返す
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Trainerが検証時に呼び出す関数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 最もスコアが高いインデックスを取得
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

def main():
    model_name = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading data...")
    train_texts, train_labels = load_sst2("../ch07/data/SST-2/train.tsv")
    dev_texts, dev_labels = load_sst2("../ch07/data/SST-2/dev.tsv")

    print("Tokenizing...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    dev_encodings = tokenizer(dev_texts, truncation=True, padding=True, max_length=128)

    # Datasetの作成
    train_dataset = SST2Dataset(train_encodings, train_labels)
    dev_dataset = SST2Dataset(dev_encodings, dev_labels)

    # モデルの準備
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="../ch09/result/",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=256,
        learning_rate=2e-5
    )

    # Trainerの初期化と実行
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics, 
    )

    # 学習開始
    print("Start Training with Trainer...")
    trainer.train()

    print("Evaluating...")
    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    main()