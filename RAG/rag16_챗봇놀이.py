'''
2026-10-07
15 카피

# rag15를 가지고 프롬프트 막 만져서 마음껏
'''

import os

from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'


# 1. 데이터

# 2. 데이터 자르기 (청킹)

# 3. 임베딩
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,
)

DB_PATH = './_db/Chroma12/'

# ========================================
# 불러오기
# ========================================
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma12',
)

# print(f"벡터 저장소에 저장된 문서 수 : : {vector_store._collection.count()}",)
# # 벡터 저장소에 저장된 문서 수 : : 52

# query = "반도체에서 눈여겨 봐야할 부분은 무엇인가요?"
# result = vector_store.similarity_search(query)         # 유사도 검색

# print(f"검색 결과의 길이 : {len(result)}")              # 4

########################### retrievers ############################
######################## 검색기(자료찾기) #########################

retriever = vector_store.as_retriever(search_kwargs={"k":2})
print(retriever)
# tags=['Chroma', 'OpenAIEmbeddings']
# vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x000001D94EBD2090> search_kwargs={'k': 2}

# aaa = retriever.invoke(query)
# print(f"검색된 관련 문서 수 : {len(aaa)}")  # 검색된 관련 문서 수 : 2
# print(f"첫번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}...")


print("=============================================================================")
######################## 모델 연결 #################################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'gpt-6-luna',
    temperature=0,     # 응답의 무작위성을 낮춤
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

# 앞에서 배웠던 # 4.predict가 여기서 invoke였다.
# response = model.invoke("반도체에서 눈여겨 봐야할 부분은 무엇인가요?")
# print("model의 답변 :", response.content)
# model의 답변 : 삼성전자는 삼성그룹의 계열사로, 창업주는 이건희의 부친인 이병철(肄炳哲, Lee Byung-chul)입니다.
#  다만 삼성그룹 자체의 창업으로 보면 이병철이 1938년에 삼성상회로 처음 시작한 뒤 삼성의 발전을 이끈 창업주로 간주됩니다.
#  이후 삼성전자는 이건희 회장 시절에 글로벌 전자 기업으로 성장했습니다.
print("=============================================================================")
print("=============================================================================")

# query_with_context = f"""
#     {aaa[0].page_content}\n\n
#     위 내용에 근거하여 다음 질문에 답변하세요. \n\n{query}
# """

# response = model.invoke(query_with_context)
# print("model의 대답 :",response.content)

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 
컨텍스트 관련 정보가 없다면 적당히 맞는 말인 것 처럼 거짓말로 지어내서 말씀해 주세요.
거짓말도 

컨텍스트 : {context}
질문 : {input}
답변 : 
""")

# 체인 만들기
docu_chain = create_stuff_documents_chain(model, prompt)   # prompt | model  = 검색된 문서들을 LLM에게 넣어서 답변을 만들게 하는 체인
rag_chain = create_retrieval_chain(retriever, docu_chain)  # 검색 | docu_chain

"""
# 체인 실행
query = "반도체는 전망이 어때요?"
response = rag_chain.invoke({"input" : query})

print('response :')
print(response)
print(" [ keys() ] ==============================================")
print(response.keys())
# dict_keys(['input', 'context', 'answer'])
print(" [ context ] ============================================")
print(response['context'][0].page_content)
print(" [ answer ] =============================================")
print(response['answer']) # 주어진 정보로는 답변할 수 없습니다.

"""
# gradio 챗봇#################################
# gradio 챗봇#################################
import gradio as gr

def answer_invoke(message, history):
    response = rag_chain.invoke({"input" : message})
    return response['answer']

# gradio 인터페이스 만들자
demo = gr.ChatInterface(fn=answer_invoke, title='응애!!')

# gradio 실행
demo.launch()





