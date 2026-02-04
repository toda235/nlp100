from transformers import pipeline

model_name = "google-bert/bert-base-uncased"
fill_mask = pipeline("fill-mask", model_name = model_name)

masked_text = "The movie was full of [MASK]."
outputs = fill_mask(masked_text) 

print(outputs)
print("=====")
print(outputs[0]['sequence'])
print(outputs[0]['token_str'])