import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)

def load_sst2(path):
    with open(path, encoding="utf-8") as f:
        data = f.read().rstrip().split("\n")
        data = [item.split("\t") for item in data[1:]]

    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]

    return sentences, labels

train_sentences, train_labels = load_sst2("../ch07/data/SST-2/train.tsv")
dev_sentences, dev_labels = load_sst2("../ch07/data/SST-2/dev.tsv")

train_dataset = Dataset.from_dict({
    "sentence": train_sentences,
    "label": train_labels
})

dev_dataset = Dataset.from_dict({
    "sentence": dev_sentences,
    "label": dev_labels
})

model_name = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # GPT-2はpad_tokenを持たないため、eos_tokenを割り当てる。

model = AutoModelForCausalLM.from_pretrained(model_name)


def preprocess(example):
    label_text = "Positive" if example["label"] == 1 else "Negative"

    prompt = f"Classify the sentiment.\n\nSentence: {example['sentence']}\n\nLabel:"
    full_text = prompt + " " + label_text

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=128,
        padding="max_length"
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # ラベル部分だけ loss 計算
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=128
    )["input_ids"]

    labels = [-100] * len(input_ids)

    for i in range(len(prompt_ids), len(input_ids)):
        labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

train_dataset = train_dataset.map(preprocess, remove_columns=["sentence", "label"])
dev_dataset = dev_dataset.map(preprocess, remove_columns=["sentence", "label"])

train_dataset.set_format(type="torch")
dev_dataset.set_format(type="torch")


training_args = TrainingArguments(
    output_dir="./result",
    eval_strategy="epoch", 
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    logging_dir="./logs",
    save_strategy="epoch",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset
)

trainer.train()

trainer.save_model("./result")
tokenizer.save_pretrained("./result")
