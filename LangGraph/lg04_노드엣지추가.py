from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()    # .strip() 공백이나 띄어쓰기 무시해줌. 오타방지
base_url = 'https://monogpt.kr/api/monorouter/v1'

llm = ChatOpenAI(
    model_name = 'gpt-6-luna',
    temperature = 0,
    max_tokens=1000,
    openai_api_key = api_key,
    base_url = base_url,
)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage

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
def prepare(state : State):
    return{"messages" :
           [AIMessage(content="질문을 접수했습니다.")]
           }

def chatbot(state : State) :
    return {"messages" : [llm.invoke(state["messages"])]}

def finish(state : State):
    return{"messages" :
           [AIMessage(content="답변 처리가 완료되었습니다.")]
           }

#*############### 3. 그래프 구성 ###################
graph_01 = StateGraph(State)

graph_01.add_node("prepare", prepare)
graph_01.add_node("chatbot", chatbot)
graph_01.add_node("finish", finish)

#*############### 4. 엣지 추가 #####################
graph_01.add_edge(START, "prepare")
graph_01.add_edge("prepare", "chatbot")
graph_01.add_edge("chatbot", "finish")
graph_01.add_edge("finish", END)

#*############### 5. 그래프 컴파일 ###################
graph = graph_01.compile()

#*############### 6. 그래프 실행 ####################
query = "삼성전자의 주가를 예측해줘"

response = graph.invoke({
            "messages" : [("user", query)]
})

#*############ 7. 결과 출력 ###############
for i, msg in enumerate(response['messages']):
    print(f"[{i}] {msg.type} :{msg.content}")

#* [0] human :삼성전자의 주가를 예측해줘
#* [1] ai :질문을 접수했습니다.
#* [2] ai :삼성전자 보통주(005930)를 기준으로 보면, **제 기본 전망은 ‘중립~조건부 긍정’**입니다. 다만 실시간 주가와 최신 실적을 확인하지 못하므로 구체적인 목표가나 단기 등락률을 제시하는 것은 신뢰하기 어렵습니다.
#* 
#* - **상승 요인:** 메모리 반도체 업황 회복, AI용 HBM 공급 확대와 수익성 개선, 파운드리 사업의 고객·수율 개선  
#* - **하락 요인:** HBM 경쟁에서의 성과 지연, 메모리 가격 둔화, 파운드리 적자 장기화, 미·중 규제와 환율 변동  
#* - **기간별 판단:** 단기에는 실적 발표와 반도체 업황 뉴스에 따라 변동성이 클 수 있습니다. 중장기에는 AI 메모리에서 실제 점유율과 이익이 늘어나는지가 핵심입니다.
#* 
#* 따라서 한마디로는 **“상승 잠재력은 있지만, HBM·파운드리 실적 확인 전까지는 신중한 접근이 필요하다”**고 봅니다. 매수 여부를 판단할 때는 최신 실적과 밸류에이션을 확인하고, 한 번에 큰 금액을 투입하기보다 분할 매수와 손실 감내 범위를 고려하세요.  
#* 원하시면 **투자 기간(예: 3개월·1년)**과 현재 주가를 알려주시면 그 기준으로 상승·기준·하락 시나리오를 더 구체적으로 정리해드릴게요.
#* [3] ai :답변 처리가 완료되었습니다.