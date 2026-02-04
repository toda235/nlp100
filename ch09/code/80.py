from transformers import AutoTokenizer

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "The movie was full of incomprehensibilities."
token = tokenizer.tokenize(text) 

print(token)