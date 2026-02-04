import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Max Pooling 
class BertWithMaxPooling(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels

    # **kwargs で他の引数も受け取れるように設定
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        # マスキング処理
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_hidden_state = last_hidden_state.masked_fill(mask == 0, -1e9)

        # Max Pooling
        pooled_output, _ = torch.max(masked_hidden_state, dim=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

def load_sst2(path):
    df = pd.read_table(path)
    return df["sentence"].tolist(), df["label"].tolist()

class SST2Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def main():
    model_name = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_texts, train_labels = load_sst2("../ch07/data/SST-2/train.tsv")
    dev_texts, dev_labels = load_sst2("../ch07/data/SST-2/dev.tsv")

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    dev_encodings = tokenizer(dev_texts, truncation=True, padding=True, max_length=128)

    train_dataset = SST2Dataset(train_encodings, train_labels)
    dev_dataset = SST2Dataset(dev_encodings, dev_labels)

    # カスタムモデルの初期化
    model = BertWithMaxPooling(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="../ch09/result_custom/",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=256,
        learning_rate=2e-5,
        remove_unused_columns=False 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics, 
    )

    print("Start Training...")
    trainer.train()

    print("Evaluating...")
    results = trainer.evaluate()
    print(f"Validation Accuracy: {results['eval_accuracy']:.4f}")

if __name__ == "__main__":
    main()