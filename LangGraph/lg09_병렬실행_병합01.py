from io import BytesIO
from typing import TypedDict

import matplotlib.pyplot as plt
from langgraph.graph import END, START, StateGraph
from PIL import Image


# * 1. Sate 정의
class State(TypedDict):
    question: str
    search_result: str
    analysis_result: str
    final_result: str


# * 2. 검색 Node
def search_agent(state: State):
    print("검색 에이전트 실행")
    return {"search_result": "엔비디아는 AI GPU 시장의 주요 기업이다."}


# * 3. 분석 Node
def analysis_agent(state: State):
    print("분석 에이전트 실행")
    return {"analysis_result": "AI 반도체 수요 증가가 성장 요인이다."}


# * 4. 결과 통합 Node
def merge_agent(state: State):
    result = state["search_result"] + "\n" + state["analysis_result"]
    return {"final_result": result}


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

# * 실행
response = graph.invoke({"question": "엔비디아의 미래 전망은?"})

print(response["final_result"])

# *##########################################
# * 그래프 이미지 저장2
# *##########################################


png_data = graph.get_graph().draw_mermaid_png()

img = Image.open(BytesIO(png_data))

# with open(path + "lg07_ToolNode.png", 'wb') as file:   #? write binary
#     file.write(png_data)

# file = open(path + "lg07_ToolNode.png", 'wb')
# file.write(png_data)
# file.close()

# * 그림출력
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis("off")
plt.show()
