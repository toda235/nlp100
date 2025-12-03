import os
from dotenv import load_dotenv
import google.generativeai as genai

# .env を読み込む
load_dotenv()

# .env からAPIキーを取得する
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
)

prompt ="""あなたは日本の川柳の専門家です。「人生」をお題として川柳を10句作ってください。"""

response = model.generate_content(prompt).text
print(response)