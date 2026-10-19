'''
#* 23_2 카피
#! 이거 안 됐음 다시 해야함
'''

from langchain_core.documents import Document

texts = [
    Document(page_content="삼성전자는 반도체를 생산하는 기업이다."),
    Document(page_content="삼성전자는 갤럭시 스마트폰을 판매한다."),
    Document(page_content="SK하이닉스는 메모리 반도체를 생산한다."),
    Document(page_content="엔비디아는 인공지능 연산용 GPU를 설계한다."),
    Document(page_content="애플은 아이폰을 개발하는 기업이다."),
    Document(page_content="테슬라는 전기자동차를 제조한다."),
    Document(page_content="삼성전자는 AI 반도체 기술을 연구한다."),
    Document(page_content="반도체는 컴퓨터의 핵심 부품이다."),
]

query = '삼성전자의 반도체 사업'

# #*################ BM25 검색 #########################
# # BM25는 키워드 기반 검색. (Best Match 25)
# from langchain_community.retrievers import BM25Retriever

# bm25_retriever = BM25Retriever.from_documents(texts)
# bm25_retriever.k = 3

# bm25_docs = bm25_retriever.invoke(query)

# #? activate langchain
# #? pip install rank_bm25
# print('================== BM25 검색 결과 =====================')
# for doc in bm25_docs:
#     print(doc.page_content)



'''
================== BM25 검색 결과 =====================
삼성전자는 AI 반도체 기술을 연구한다.
반도체는 컴퓨터의 핵심 부품이다.
테슬라는 전기자동차를 제조한다.

'''

#*################ 벡터 검색 #########################
import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

# pip install faiss-cpu
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'


#03. 임베딩
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import sentence_transformers

embeddings = HuggingFaceEmbeddings(
    model_name = "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs ={
        "device" : "cpu",
        "local_files_only" : True
    } 
)


DB_PATH = './_db/Faiss20'
# db.save_local(
#     folder_path = DB_PATH,
#     index_name='faiss_index17'
# )

db = FAISS.from_documents(
    index_name='20qwen_pdf',
    embedding=embeddings,
    persist_directory=DB_PATH,
    documents=texts,
)

db_retriever = db.as_retriever(
    search_kwargs={"k":3}
)
db_docs = db_retriever.invoke(query)

# print('================== 벡터독스 검색 결과 =====================')
for doc in db_docs:
    print(doc.page_content)




#* ############################################
#* 하이브리드 검색 ############################
from langchain_classic.retrievers import EnsembleRetriever

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        vector_retriever,
    ],
    weights=[0.5, 0.5]
)

hybrid_docs = hybrid_retriever.invoke(query)

# print('================== 하이브리드 검색 결과 =====================')
for doc in hybrid_docs:
    print(doc.page_content)