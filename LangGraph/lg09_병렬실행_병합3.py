import os
from io import BytesIO
from typing import Annotated, TypedDict

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from PIL import Image

load_dotenv()

api_key = os.environ[
    "MONOROUTER_API_KEY"
].strip()  # .strip() 공백이나 띄어쓰기 무시해줌. 오타방지
base_url = "https://monogpt.kr/api/monorouter/v1"

llm = ChatOpenAI(
    model_name="gpt-5.6-terra",
    temperature=0,
    max_tokens=2000,
    openai_api_key=api_key,
    base_url=base_url,
)

# pip install langchain-tavily
# * ########################### Tool 정의 ###########################

# * 웹 검색 도구 생성
web_search = TavilySearch(
    max_results=5,
)

# result = web_search.invoke({"query": "삼성전자 주가"})

# print(result)
# print(result['results'][-1]['content'])

tools = [web_search]

llm_with_tools = llm.bind_tools(tools)


# *################## 1. State 정의 ####################
class State(TypedDict):  # typeddict를 상속받아서 마음대로 써도 된다.
    messages: Annotated[list, add_messages]
    search_result: str
    analysis_result: str
    # final_result: str


# *################## 2. Node 정의 ####################
# * 검색 Agent
def search_agent(state: State):
    print("검색 Agent 실행")
    question = state["messages"][-1].content

    result = web_search.invoke({"query": question})
    return {"search_result": str(result)}


def analysis_agent(state: State):
    print("분석 Agent 실행")
    question = state["messages"][-1].content

    response = llm.invoke(f"다음 질문을 기존 지식을 바탕으로 분석해줘.\n{question}")
    return {"analysis_result": response.content}


def merge_agent(state: State):  # ? 결과 통합 에이전트
    print("결과 통합 에이전트 실행")

    prompt = f"""
    다음 두 결과를 종합하여 최종 답변을 작성해줘.

    [웹 검색 결과]
    {state["search_result"]}

    [AI분석 결과]
    {state["analysis_result"]}

    검색 결과를 우선적으로 참고하고,
    최신 정보와 기존 분석을 구분해서 설명해줘.
    """
    response = llm.invoke(prompt)
    return {"messages": [response]}


# *############ 3. Graph 구성 ###################
# * Graph 구성
builder = StateGraph(State)

builder.add_node("search", search_agent)
builder.add_node("analysis", analysis_agent)
builder.add_node("merge", merge_agent)

# *############# 병렬 실행 #############
builder.add_edge(START, "search")
builder.add_edge(START, "analysis")

# *############# 병합 ###############
builder.add_edge("search", "merge")
builder.add_edge("analysis", "merge")

builder.add_edge("merge", END)

# * 컴파일
graph = builder.compile()

# *############# 그래프 실행 #################
# * 실행
query = "엔비디아의 미래 전망에 대해 알려줘."

response = graph.invoke(
    {"messages": [("user", query)]},
)
print("Assistant :", response["messages"][-1].content)

# *##########################################
# * 그래프 이미지 저장2
# *##########################################


png_data = graph.get_graph().draw_mermaid_png()

img = Image.open(BytesIO(png_data))

# * 그림출력
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis("off")
plt.show()
