"""
#! tool calling : 외부 기능을 가져와서 사용하기!!
"""

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()    # .strip() 공백이나 띄어쓰기 무시해줌. 오타방지
base_url = 'https://monogpt.kr/api/monorouter/v1'

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature = 0,
    max_tokens=3000,
    openai_api_key = api_key,
    base_url = base_url,
)

# pip install langchain-tavily
#* ########################### Tool 정의 ###########################
from langchain_tavily import TavilySearch

# #* 웹 검색 도구 생성
web_search1 = TavilySearch(
    max_results=5,
)

# pip install ddgs
#* ########################### Tool 정의 ###########################
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

# #* 웹 검색 도구 생성
web_search2 = DuckDuckGoSearchRun(
    max_results=5,
)

# result = web_search.invoke({
#     "query" : "삼성전자 주가"
# })

# print(result)
# print(result['results'][-1]['content'])

# exit()

tools = [web_search1, web_search2]

llm_with_tools = llm.bind_tools(tools)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

#*################## 1. State 정의 ####################
class State(TypedDict) :  # typeddict를 상속받아서 마음대로 써도 된다.
    messages : Annotated[list, add_messages]

#*################## 2. Node 정의 ####################
def chatbot(state : State):             #?state = State()  # << 이걸 받아들인다는 뜻
    return{"messages":[llm_with_tools.invoke(state["messages"])]}

#*########### 3. Graph 구성, Node 추가 #################
#* 그래프 생성
graph_builder = StateGraph(State)       #? 상속받은 스테이트 그래프를 어쩌구 저쩌구

graph_builder.add_node("chatbot", chatbot)

# graph_builder.add_node("tools", ToolNode(tools))
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)


#*################## 4. Edge 정의 #####################
graph_builder.add_edge(START, "chatbot")

# graph_builder.add_edge("chatbot", "tools")
graph_builder.add_conditional_edges(    #? 조건부 엣지
    "chatbot",
    tools_condition       #? tool_calls 이 있냐?
    #* tool_calls가 있으면 'tools' 반환
    #* tool_calls가 없으면 END 반환
    #* tool_condition에 END가 포함되어 있음.
)

graph_builder.add_edge("tools", "chatbot")

#*################## 5. 컴파일 #####################
graph = graph_builder.compile()

#*################## 6. 그래프 실행 #####################
query = "삼성전자의 오늘 주가 알려줘"

response = graph.invoke({
    "messages" : [("user", query)]
})

print("AI : ", response["messages"][-1].content)


