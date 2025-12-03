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

prompt ="""あなたは日本の川柳の評価者です。以下に「人生」をお題として川柳を10句示すので、これらの川柳の面白さを10段階してください。

1.  産声は ゴールテープの スタートだ
2.  迷子札 つけたつもりが 迷子なり
3.  回り道 それも景色と 言い聞かせ
4.  夕焼けに 染まる背中と 来し方よ
5.  宝物 古傷笑窪 増えていく
6.  定年後 第二の青春 登山道
7.  人生は 喜劇と悲劇 玉手箱
8.  ありがとう 言えるうちにね 今すぐに
9.  遺影には 最高の笑顔 飾ろうか
10. 種まけば いつか花咲く 人生道

"""

response = model.generate_content(prompt).text
print(response)