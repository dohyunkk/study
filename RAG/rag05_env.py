'''
2029-09-30

API키는
1. 코드에 직접 입력                      # 노출위험도 가장 높음 
2. 내 컴퓨터 환경변수에 저장하고 불러오기  
3. env파일을 만들어서 저장하고 불러오기
중 3번

.env 파일을 만들어주고 load_dotenv로 가져올 수 있다.
현재 작업중인 그룹(폴더)의 env를 가져온다.

from dotenv import load_dotenv
load_dotenv()

'''

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature = 0,
    # openai_api_key = openai_api_key,
)

response = llm.invoke('나는 세계1짱 김왕장. 내가 누구?' )
# print(response)
print(response.content)
# 세계 1짱, 김왕장님이십니다. 👑