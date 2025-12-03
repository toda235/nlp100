import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
import re

# .env を読み込む
load_dotenv()

# .env からAPIキーを取得する
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config={
        "temperature":0.7,
        "top_p":0.9,
    }
)

df = pd.read_csv("data/JMMLU/JMMLU/global_facts.csv")
df.columns = ["question","A","B","C","D","answer"] 
total = len(df)
correct = 0

prompt = """あなたは世界事実の専門家です。この中から適切な回答を選択肢A,B,C,Dの中から選択してください。"""


for index, row in df.iterrows():
    question = row["question"]
    # answer = row["answer"]
    # new_answer = row[answer]
    options = [row["A"], row["B"], row["C"], row[row["answer"]]]

    question_prompt = f"""{prompt}
    次の世界事実の問題を解いてください。
    
    問題: {question}
    
    選択肢:
    A. {options[0]}
    B. {options[1]}
    C. {options[2]}
    D. {options[3]}
    
    回答は一つだけ選んでください。理由や説明は不要です。
    """
    response = model.generate_content(question_prompt)
    generated_answer = response.text.strip().upper()
    
    if "A" in generated_answer:
        model_answer = "A"
    elif "B" in generated_answer:
        model_answer = "B"
    elif "C" in generated_answer:
        model_answer = "C"
    elif "D" in generated_answer:
        model_answer = "D"  
    else:
        model_answer = None
    
    if model_answer == "D":
        correct += 1
            
accuracy = correct / total
        
print(f"Correct: {correct}/{total}")
print(f"Accuracy: {accuracy:.2%}")            