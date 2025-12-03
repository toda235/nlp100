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
    model_name="gemini-2.5-flash",
)

df = pd.read_csv("data/JMMLU/JMMLU/global_facts.csv")
df.columns = ["question","A","B","C","D","answer"] 
total = len(df)
correct = 0

prompt = """あなたは世界事実の専門家です。この中から適切な回答を選択肢A,B,C,Dの中から選択してください。"""

for index, row in df.iterrows():
    question = row["question"]
    options = [row["A"], row["B"], row["C"], row["D"]]
    answer = row["answer"]

    question_prompt = f"""{prompt}
    次の世界事実の問題を解いてください。
    
    問題: {question}
    
    選択肢:
    A. {options[0]}
    B. {options[1]}
    C. {options[2]}
    D. {options[3]}
    
    回答は一つだけ、A,B,C,Dの中から選んでください。
    """
    response = model.generate_content(question_prompt)
    generated_answer = response.text.strip()
    
    #　文章での回答の除去
    match = re.search(r'[A-D]', generated_answer)
    if match:
        model_answer = match.group(0)
        
        if model_answer == answer:
            correct += 1
            
accuracy = correct / total
        
print(f"Correct: {correct}/{total}")
print(f"Accuracy: {accuracy:.2%}")            