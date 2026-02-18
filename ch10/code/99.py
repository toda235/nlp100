import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer, DPOConfig

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
# GPT-2はpad_tokenを持たないため、eos_tokenを割り当てる。
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# DPOでは、Policyモデル(学習対象)とReferenceモデル(比較対象)が必要だが、DPOTrainerにmodelのみを渡すと、自動的にコピーしてReferenceモデルを作成する。
# DPOでは {prompt, chosen, rejected} の形式が必要

def dpo_preprocess(example):
    # プロンプトの作成
    prompt = f"Classify the sentiment.\n\nSentence: {example['sentence']}\n\nLabel:"
    
    # ラベルに応じた 正解(chosen) と 不正解(rejected) の作成
    if example["label"] == 1:
        chosen = " Positive"
        rejected = " Negative"
    else:
        chosen = " Negative"
        rejected = " Positive"

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

train_dataset = train_dataset.map(dpo_preprocess, remove_columns=["sentence", "label"])
dev_dataset = dev_dataset.map(dpo_preprocess, remove_columns=["sentence", "label"])

training_args = DPOConfig(
    output_dir="./result_dpo",
    eval_strategy="epoch",
    num_train_epochs=1,          
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=32,
    logging_dir="./logs_dpo",
    save_strategy="epoch",
    fp16=True,
    learning_rate=5e-5,          
    beta=0.1,                    
    remove_unused_columns=False  
)

trainer = DPOTrainer(
    model=model,
    ref_model=None, # modelをコピーしてref_modelとして使用
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    processing_class=tokenizer,
)

trainer.train()

trainer.save_model("./result_dpo")
tokenizer.save_pretrained("./result_dpo")