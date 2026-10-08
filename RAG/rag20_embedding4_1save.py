import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

# =========================================
# 1. 데이터(pdf) 가져오기
path = './_data/'
pdf_loader = PyPDFLoader(path + 'attention is all you need.pdf')
pdf_docs = pdf_loader.load()

# 문서 개수 확인
# print(type(pdf_docs)) # <class 'list'>
# print(len(pdf_docs))  # 15  # 15page
# print(pdf_docs)

print("=============================")
# print(pdf_docs[0])

# exit()

# =========================================
# 2.  데이터 자르기 (청킹)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
)
split_docs = splitter.split_documents(pdf_docs)


# =========================================
# 3. 임베딩
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import sentence_transformers

# embeddings = HuggingFaceEmbeddings(
#     model_name = 'BAAI/bge-m3',
#     model_kwargs ={
#         "device" : "cpu",
#         # "local_files_only" : True
#     } 
# )

embeddings = HuggingFaceEmbeddings(
    model_name = "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs ={
        "device" : "cpu",
        # "local_files_only" : True
    } 
)

# =========================================
# 4. FAISS에 저장
db = FAISS.from_documents(split_docs, embeddings)
db.save_local("./_db/Faiss20", index_name="20qwen_pdf")