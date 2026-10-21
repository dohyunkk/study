"""
21 카피

리스트는 이터레이터다
이터레이터는 포문(반복문)이다
"""

import os

import gradio as gr
from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# pip install faiss-cpu

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"


# 03. 임베딩

embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device": "cpu", "local_files_only": True},
)


DB_PATH = "./_db/Faiss20"
# db.save_local(
#     folder_path = DB_PATH,
#     index_name='faiss_index17'
# )

db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name="20qwen_pdf",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

print(f"저장된 문서 수 : {len(db.index_to_docstore_id)}")


# exit()
########################### retrievers ############################
######################## 검색기(자료찾기) #########################

retriever = db.as_retriever(search_kwargs={"k": 2})  # * 청킹을 2개
print(retriever)

print("=============================================================================")
######################## 모델 연결 #################################

model = ChatOpenAI(
    model="gpt-6-luna",
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

print("=============================================================================")
print("=============================================================================")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 컨텍스트 관련 정보가 없다면,
"주어진 정보로는 답변할 수 없습니다."라고 말씀해 주세요.

컨텍스트 : {context} 
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# 체인 만들기
docu_chain = create_stuff_documents_chain(
    model, prompt
)  # prompt | model  = 검색된 문서들을 LLM에게 넣어서 답변을 만들게 하는 체인
rag_chain = create_retrieval_chain(retriever, docu_chain)  # 검색 | docu_chain


############## gradio 챗봇#################################
############## gradio 챗봇#################################


def answer_invoke(message, history):
    chat_history = []  # chat_history 생성

    for item in history:
        content = item["content"]

        if item["role"] == "user":
            chat_history.append(HumanMessage(content=content))
        elif item["role"] == "assistant":
            chat_history.append(AIMessage(content=content))

    response = rag_chain.invoke({"input": message, "chat_history": chat_history})

    # ? 터미널 상으로 실제 검색된 문서 2개 확인.
    for i, doc in enumerate(response["context"], start=1):
        print(f"\n검색 문서 {i}")
        print("출처:", doc.metadata.get("source"))
        print("내용:", doc.page_content)
        print("============================================")

    return response["answer"]


# gradio 인터페이스 만들자
demo = gr.ChatInterface(fn=answer_invoke, title="슛!!")

# gradio 실행
demo.launch()


"""
1. 안녕
history = [
    {"role" : "user", "content" : "안녕"}
    {"role" : "assistant", "content" : "안녕하세요"} 
]

2. 나는 윤영선이다.
history = [
    {"role" : "user", "content" : "안녕"}
    {"role" : "assistant", "content" : "안녕하세요"}
    {"role" : "user", "content" : "나는 윤영선이다."}
    {"role" : "assistant", "content" : "당신은 윤영선이군요"}
] 

3. 내가 누구게.
history = [
    {"role" : "user", "content" : "안녕"}
    {"role" : "assistant", "content" : "안녕하세요"}
    {"role" : "user", "content" : "나는 윤영선이다."}
    {"role" : "assistant", "content" : "당신은 윤영선이군요"}
    {"role" : "user", "content" : "내가 누구게?"}
    {"role" : "assistant", "content" : "선생님은 윤영선~"}
] 
"""
