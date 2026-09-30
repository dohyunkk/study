'''
2029-09-30
API키는
1. 코드에 직접 입력                      # 노출위험도 가장 높음 
2. 내 컴퓨터 환경변수에 저장하고 불러오기  
3. env파일을 만들어서 저장하고 불러오기
중 1번
현업(회사)에서 이렇게(api key 직접 입력) 하면 난리난다. (키가 노출되기 때문)
노출되면 ㅈ댐.
하지만 그냥 해본다. (가능하다는 것만 확인.)

장기적인 작업은 힘들다. 어짜피 알 거 없다. 쓸 일이 없다.
'''

from langchain_openai import ChatOpenAI
import os
os.environ['OPENAI_API_KEY'] = 'fgh-GmsldfghdfghE-1ASxDz0rGT3dfghfgfgh'

# openai_api_key = 'fgh-GmsldfghdfghE-1ASxDz0rGT3dfghfgfgh'

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature = 0,
    # openai_api_key = openai_api_key,
)

response = llm.invoke('나는 세계1짱 김왕장. 내가 누구?' )
# print(response)
print(response.content)
# 세계 1짱, 김왕장님이십니다. 👑