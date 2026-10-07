'''
2026-10-07
12-3 카피
save 데이터를 불러와서 출력값 확인.
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
# 벡터 저장소에 저장된 문서 수 : : 52

query = "반도체에서 눈여겨 봐야할 부분은 무엇인가요?"
result = vector_store.similarity_search(query)

print(f"검색 결과의 길이 : {len(result)}")              # 4

########################### retrievers ############################
############################# 검색기 ##############################

retriever = vector_store.as_retriever(search_kwargs={"k":2})
print(retriever)
# tags=['Chroma', 'OpenAIEmbeddings']
# vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x000001D94EBD2090> search_kwargs={'k': 2}

aaa = retriever.invoke(query)
print(f"검색된 관련 문서 수 : {len(aaa)}")  # 검색된 관련 문서 수 : 2
print(f"첫번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}...")


print("=============================================================================")
######################## 모델 연결 #################################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'gpt-5-nano',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

# 앞에서 배웠던 # 4.predict가 여기서 invoke였다.
response = model.invoke("반도체에서 눈여겨 봐야할 부분은 무엇인가요?")
print("model의 답변 :", response.content)
# model의 답변 : 삼성전자는 삼성그룹의 계열사로, 창업주는 이건희의 부친인 이병철(肄炳哲, Lee Byung-chul)입니다.
#  다만 삼성그룹 자체의 창업으로 보면 이병철이 1938년에 삼성상회로 처음 시작한 뒤 삼성의 발전을 이끈 창업주로 간주됩니다.
#  이후 삼성전자는 이건희 회장 시절에 글로벌 전자 기업으로 성장했습니다.
print("=============================================================================")
print("=============================================================================")

query_with_context = f"""
    {aaa[0].page_content}\n\n
    위 내용에 근거하여 다음 질문에 답변하세요. \n\n{query}
"""

response = model.invoke(query_with_context)
print("model의 대답 :",response.content)