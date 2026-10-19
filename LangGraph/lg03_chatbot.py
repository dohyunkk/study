from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()    # .strip() 공백이나 띄어쓰기 무시해줌. 오타방지
base_url = 'https://monogpt.kr/api/monorouter/v1'

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature = 0,
    max_tokens=1000,
    openai_api_key = api_key,
    base_url = base_url,
)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


#*################# 1. State 정의 #####################
class State(TypedDict):
    messages : Annotated[list, add_messages]     # 깔끔하게 들어가는 랭그래프 문법이구나 알고만 있자.
    aaa : str
    bbb : str
    ccc : int

#? 위에꺼는 한 마디로 아래다!
# state = {
# "messages" : [],               # State 상태
# "aaa" : 'abc'
# "bbb" : 'bcd',
# "ccc" : '123'
# }


#*############### 2. Node 정의#####################
def chatbot(state : State) :
    return {"messages" : [llm.invoke(state["messages"])]}


#*############### 3. 그래프 구성 ###################
graph_01 = StateGraph(State)

graph_01.add_node("chatbot",chatbot)

#*############### 4. 엣지 추가 #####################
graph_01.add_edge(START, "chatbot")

graph_01.add_edge("chatbot", END)


#*############### 5. 그래프 컴파일 ###################
graph = graph_01.compile()

#*############### 6. 그래프 실행 ####################
query = "삼성전자의 주가를 예측해줘"

response = graph.invoke({
            "messages" : [("user", query)]
})

print(response)
# print(response["messages"][0])
# print(response["messages"][1])
print(response["messages"][0].content)
print(response["messages"][1].content)
print(response["messages"][0].type)
print(response["messages"][1].type)



#* {'messages': [HumanMessage(content='삼성전자의 주가를 예측해줘', additional_kwargs={}, response_metadata={}, id='9d3b65b5-98fb-4395-a5ae-2980b623378a')]}