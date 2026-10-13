import os

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 저장할 때 사용한 경로와 인덱스 이름
DB_PATH = './_db/Faiss20'
INDEX_NAME = '20qwen_pdf'


# 저장할 때 사용한 임베딩 모델과 동일하게 설정
embeddings = HuggingFaceEmbeddings(
    model_name='Qwen/Qwen3-Embedding-0.6B',
    model_kwargs={
        'device': 'cpu',
        'local_files_only': True,
    },
)


# FAISS 저장소 불러오기
db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name=INDEX_NAME,
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)


# 폴더와 저장된 청크 개수 확인
print('저장 폴더 안의 파일:', os.listdir(DB_PATH))
print('저장된 청크 수:', len(db.index_to_docstore_id))


# 저장된 모든 청크의 본문과 메타데이터 출력
for 번호, (faiss_id, doc_id) in enumerate(
    db.index_to_docstore_id.items(),
    start=1,
):
    doc = db.docstore.search(doc_id)

    print(f'\n========== 청크 {번호} ==========')
    print('출처:', doc.metadata.get('source'))
    print('페이지:', doc.metadata.get('page'))
    print('내용:')
    print(doc.page_content[:500])  # 앞에서 500글자만 출력


# 질문과 관련된 청크만 검색해 보고 싶으면 아래 주석을 풀어 실행
# query = '질문을 여기에 입력하세요'
# results = db.similarity_search(query, k=2)
#
# print('\n========== 유사도 검색 결과 ==========')
# for 번호, doc in enumerate(results, start=1):
#     print(f'\n검색 문서 {번호}')
#     print('출처:', doc.metadata.get('source'))
#     print('페이지:', doc.metadata.get('page'))
#     print(doc.page_content)