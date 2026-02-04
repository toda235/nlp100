from transformers import pipeline

model_name = "google-bert/bert-base-uncased"
fill_mask = pipeline("fill-mask", model = model_name, top_k = 10)

masked_text = "The movie was full of [MASK]."
   
outputs = fill_mask(masked_text) 

for i in range(10):
    print(f'[MASK]:{outputs[i]['token_str']}')
    print(f'score:{outputs[i]['score']}')
    print("---")