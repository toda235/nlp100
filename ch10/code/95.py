from transformers import AutoTokenizer,AutoModelForCausalLM
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto")

chat = [
    {"role": "system", "content": "You are an excellent assistant."},
    {"role": "user", "content": "What do you call a sweet eaten after dinner?"}
]

prompt = tokenizer.apply_chat_template(
    chat,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
assistant_reply=tokenizer.decode(outputs[0], skip_special_tokens=True)

print(assistant_reply)

#94からの追加要素
chat.append({"role": "assistant", "content": assistant_reply})
chat.append({
    "role": "user",
    "content": "Please give me the plural form of the word with its spelling in reverse order."
})


final_prompt = tokenizer.apply_chat_template(
    chat,
    tokenize=False,
    add_generation_prompt=True
)

print(final_prompt)

# 再生成
inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
       **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
second_reply = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("\n=== 追加結果===")
print(second_reply)
