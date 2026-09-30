'''
2026-09-30

prompt = PromptTemplate.from_template("{topic}에 대해 쉽게 {how} 설명해주세요.")

input = {"topic" : "양자컴퓨터 학습 원리", "how" : "초등학생도 이해하기 쉽게"}

그리고 모델도 바꿔서 출력해보자. 모델마다 다르게 말을 하는지 확인해보시오.
'''

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()    # .strip() 공백이나 띄어쓰기 무시해줌. 오타방지
base_url = 'https://monogpt.kr/api/monorouter/v1'

prompt = PromptTemplate.from_template("{topic}에 대해 쉽게 {how} 설명해주세요.")

model = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature = 0,
    openai_api_key = api_key,
    base_url = base_url,
)

chain = prompt | model

input = {"topic" : "양자컴퓨터 학습 원리", "how" : "초등학생도 이해할 수 있도록"}

response = chain.invoke(input)   # 체인에 인풋을 불러줘

# print(response)          # 답변 내용과 함께 응답 ID, 토큰 사용량 등의 부가 정보도 출력.
print(response.content)  # content 하면 답변 내용만 꺼내서 출력