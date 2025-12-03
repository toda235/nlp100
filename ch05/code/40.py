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

prompt = """あなたは日本史の専門家です。
9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。"""
response = model.generate_content(prompt)
print(response.text)

