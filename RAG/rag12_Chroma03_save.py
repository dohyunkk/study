'''
2026-10-06
11-1 카피

text 문서가 여러개 있을 때 한 번에 땡겨오기.

from glob import glob

청킹과 토큰 문자 정리 한 번 해라.
명확한 토큰의 기준은 없다. 통상적으로 지금은 한 문자당 토큰 하나라고 인지하고 있자.

매서드가 나오면 아 함수구나~ 매서드는 특히 반복되는 것에서 많이 쓰인다.

chroma 실습 문서 저장, 실행
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

from glob import glob

path = './_data/rag_data/'

# 폴더에서 text 파일 목록 가져오기
txt_files = glob(os.path.join(path, '*.txt'))

# print(txt_files)
# ['./_data/rag_data\\2026_AI_for_All.txt', './_data/rag_data\\nvidia_outlook.txt', './_data/rag_data\\samsung_outlook.txt']


# exit()

# 01. 데이터 불러온다
data = []
for text_file in txt_files:
    loader = TextLoader(text_file, encoding='utf-8')
    # data.append(loader)
    data += loader.load()

print("==========================================")
# print(data[0])
# print("==========================================")
# print(len(data))                                             # 3
# print(data[0].page_content)

char_count = [len(doc.page_content) for doc in data]           # 간략화된 for 문
print(char_count)                                              # [8158, 2049, 1898]


# 02. 문서 자른다. (청킹) - 벡터화 하기 위해.
text_splitter = RecursiveCharacterTextSplitter(                 # 재귀적 문자 텍스트 분할기
    chunk_size=300,
    chunk_overlap=10,
    separators=["\n\n", "\n", " ", ""]                          # 통상 디폴트
)

texts = text_splitter.split_documents(data)
# print("생성된 텍스트 청크 수 :", len(texts))                           
# # 생성된 텍스트 청크 수 : 52
# print("각 청크의 길이 :", list(len(text.page_content)for text in texts))
# 각 청크의 길이 : 
# [259, 282, 282, 128, 276, 214, 158, 249, 262, 291, 268, 182, 286, 295, 182, 162, 283, 286, 257, 235, 214, 258, 207, 286, 220, 198,
#  271, 57, 272, 122, 9, 269, 299, 284, 289, 209, 222, 230, 254, 249, 296, 90, 247, 243, 185, 219, 239, 235, 298, 282, 187, 249]

# separators=["\n\n", "\n", " ", ""] ex)
# 구분자 기준에 맞게 최대한 자연스럽게 자르다 보면 300자 이 전에 잘리기도 한다.
# • 280자 지점에서 문장이 끝났고, 다음 문장까지 합치면 330자가 된다면?
# • 300자를 넘기지 않기 위해 280자에서 청크를 끊어버립니다. 그래서 300자보다 작은 200대 후반 청크들이 자주 발생합니다.

# chunk_overlap: 청크와 청크 사이에 겹치는 구간을 만들어, 문맥이 중간에 잘려도 의미가 이어지도록 도와줍니다.

print("첫번째 청크의 내용 : ", texts[0])
print("첫번째 청크의 내용 : ", texts[0].page_content)
print("첫번째 청크의 내용 : ", len(texts[0].page_content))

print("두번째 청크의 내용 : ", texts[1])
print("두번째 청크의 내용 : ", texts[1].page_content)
print("두번째 청크의 내용 : ", len(texts[1].page_content))


# page_content='2026년 한국의 AI for All 프로젝트와 생성형 AI 서비스 확산
# metadata={'source': './_data/rag_data\\2026_AI_for_All.txt'}
# 청킹돼서 나온 다큐먼츠(page_content) 안에는 컨텐츠와 메타데이타(metadata)가 있다.



from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,                                  # 출력 벡터 차원을 5 로 지정 (text-embedding-3 계열 모델에서만 사용 가능)
)

sample_text = "삼성전자의 창업자는 누구인가요?"
vector = embeddings.embed_query(sample_text)
# print(vector)
# print(len(vector))  # vector의 길이(개수) : 1536


DB_PATH = './_db/Chroma12/'

# 저장
vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
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