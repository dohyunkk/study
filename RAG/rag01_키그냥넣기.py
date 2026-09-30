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

실행할 때 마다 크레딧이 소모된다.
'''

from langchain_openai import ChatOpenAI

openai_api_key = 'sasdfasdfasdfgfhfghgfh'  # 깃허브 등 유출주의

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature = 0,
    openai_api_key = openai_api_key,
)

response = llm.invoke('나는 세계1짱 김왕장. 내가 누구?' )
# print(response)
print(response.content)
# 세계 1짱, 김왕장님이십니다. 👑