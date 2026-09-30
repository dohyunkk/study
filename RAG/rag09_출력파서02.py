'''
2026-09-30

parser : 분석하다. 답변을 다듬다.

'''
template = """
당신은 영어를 가르치는 10년차 영어 선생님입니다.
주어진 상황에 맞는 영어 회화에 맞는 영어회화를 작성해주세요.
양식은 [FORMAT]을 참고하여 작성해주세요.

#상황:
{question}

#FORMAT:
-영어회화 :
-한글번역 :
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv


load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()    # .strip() 공백이나 띄어쓰기 무시해줌. 오타방지
base_url = 'https://monogpt.kr/api/monorouter/v1'

prompt = PromptTemplate.from_template(template)

model = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature = 0,
    openai_api_key = api_key,
    base_url = base_url,
)

from langchain_core.output_parsers import StrOutputParser  # 아웃풋을 문자열로 빼주겠다.
output_parser = StrOutputParser()

chain = prompt | model | output_parser

input = {"question" : "저는 부산에서 물밀면을 먹고싶어요"}

response = chain.invoke(input)   # 체인에 인풋을 불러줘

print(response)          # 답변 내용과 함께 응답 ID, 토큰 사용량 등의 부가 정보도 출력.
# print(response.content)  # content 하면 답변 내용만 꺼내서 출력

# -영어회화 :  
# A: I’d like to try mul milmyeon in Busan. Do you know a good restaurant?  
# B: Yes, there’s a famous place nearby. Their mul milmyeon is really refreshing.  
# A: Great! I’d like a bowl of mul milmyeon, please.  
# B: Would you like it spicy?  
# A: A little spicy, please.  

# -한글번역 :  
# A: 부산에서 물밀면을 먹어 보고 싶어요. 좋은 식당을 아세요?  
# B: 네, 근처에 유명한 곳이 있어요. 그곳 물밀면은 정말 시원하고 맛있어요.  
# A: 좋아요! 물밀면 한 그릇 주세요.  
# B: 맵게 드릴까요?  
# A: 조금 맵게 해 주세요.