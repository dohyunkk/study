'''
2029-09-30

API키는
1. 코드에 직접 입력                      # 노출위험도 가장 높음 
2. 내 컴퓨터 환경변수에 저장하고 불러오기  
3. env파일을 만들어서 저장하고 불러오기
중 2번

내 API KEY를 코드에 노출하지 않기 위해, 그리고 할 때마다 불러오지 않기 위해.(번거로움 감소)
내 컴퓨터의 환경변수에 API KEY를 지정해서 불러와보자.

하단 윈도우 검색창에 (시스템 환경 변수 편집) 
-> 환경변수 탭
-> 사용자 변수 새로 만들기

하지만 물리적으로 내 컴퓨터에 접근해서 키를 가져갈 수 있음.
'''

from langchain_openai import ChatOpenAI
import os
# os.environ['OPENAI_API_KEY'] = 'fgh-GmsldfghdfghE-1ASxDz0rGT3dfghfgfgh'

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature = 0,
    # openai_api_key = openai_api_key,
)

response = llm.invoke('나는 세계1짱 김왕장. 내가 누구?' )
# print(response)
print(response.content)
# 세계 1짱, 김왕장님이십니다. 👑