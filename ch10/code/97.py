import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class SST2Dataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sentences[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    

class GPTSentimentClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state  # (B, T, H)

        # ===== Mean Pooling =====
        mask = attention_mask.unsqueeze(-1)
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.sum(mask, dim=1)
        sentence_vector = summed / counts

        logits = self.classifier(sentence_vector)
        return logits


train_df = pd.read_table("../ch07/data/SST-2/train.tsv")
dev_df = pd.read_table("../ch07/data/SST-2/dev.tsv")

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT2はpad_tokenがないので追加
tokenizer.pad_token = tokenizer.eos_token

train_dataset = SST2Dataset(
    train_df["sentence"].tolist(),
    train_df["label"].tolist(),
    tokenizer
)

dev_dataset = SST2Dataset(
    dev_df["sentence"].tolist(),
    dev_df["label"].tolist(),
    tokenizer
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=256)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPTSentimentClassifier(model_name).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

model.eval()
preds = []
gold_labels = []

with torch.no_grad():
    for batch in dev_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        predictions = torch.argmax(logits, dim=1)

        preds.extend(predictions.cpu().tolist())
        gold_labels.extend(labels.cpu().tolist())

accuracy = accuracy_score(gold_labels, preds)
print(f"Dev Accuracy: {accuracy:.4f}")