from transformers import pipeline

model_name = "openai-community/gpt2"
generator = pipeline("text-generation", model = model_name,pad_token_id=50256)

text = "The movie was full of"

for i in [0.2, 0.4, 0.6, 0.8,1.0]:
    output = generator(
        text,
        temperature=i,
        do_sample = True,
        max_new_tokens = 50
    )
    print(f"temperature:{i}:{output[0]["generated_text"]}")