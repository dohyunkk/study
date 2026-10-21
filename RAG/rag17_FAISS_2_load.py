"""
17_1 카피

"""

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# pip install faiss-cpu

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"


# 03. 임베딩

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,                  # 1536
)


DB_PATH = "./_db/Faiss17"
# db.save_local(
#     folder_path = DB_PATH,
#     index_name='faiss_index17'
# )

db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name="faiss_index17",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

print("==========================================")
# 문서 저장 ID 확인
print(db.index_to_docstore_id)
print("==========================================")
# 저장된 결과 확인
print(db.docstore._dict)
print("==========================================")
# 유사도 검색
aaa = db.similarity_search("삼성전자 창업주에 대해 알려줘.", k=2)
print(aaa)
