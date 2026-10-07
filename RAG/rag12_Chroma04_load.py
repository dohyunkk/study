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
    # dimensions=5,                                  # 출력 벡터 차원을 5 로 지정 (text-embedding-3 계열 모델에서만 사용 가능)
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

query = " 삼성전자의 창업자는 누구인가요?"
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