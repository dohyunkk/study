'''
2026-10-06
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

# ========================================================================
# 01. 데이터 불러온다
# path = "./_data/rag_data/"
# loader1 = TextLoader(path + "samsung_outlook.txt", encoding='utf8')
# loader2 = TextLoader(path + "nvidia_outlook.txt", encoding='utf8')

# 02. 문서 자른다. (청킹) - 벡터화 하기 위해.
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=300,
#     chunk_overlap=100,
#     separators=["\n\n", "\n", " ", ""]                      # 통상 디폴트
# )

# split_doc1 = loader1.load_and_split(text_splitter)          # 청크 300, 오버랩 100
# split_doc2 = loader2.load_and_split(text_splitter)          # 청크 300, 오버랩 100

# 문서 개수 확인
# print(split_doc1)
# print(split_doc2)

# print(len(split_doc1), len(split_doc2))                    # 9 9

# exit()
# ==========================================================================

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,                                  # 출력 벡터 차원을 5 로 지정 (text-embedding-3 계열 모델에서만 사용 가능)
)

DB_PATH = './_db/Chroma11/'

# 저장
db = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma11',
)

# 저장된 데이터 확인
# print("============================================")
# print("db :")
# print(db.get())   

# db :
# {'ids': 
# ['dcb3956b-044d-4ed2-8c41-b9320849297c', '5fbb88dc-7d66-4297-8eb5-99cf565158ad', 'e79e4be4-6d26-4bf9-bb7b-20e724e2f997',
#  '269182e8-f1d3-46a0-a99f-173db49d87c6', '8591c012-fd4f-4c57-9022-04434a000c31', '7926d4d0-35f8-4906-bc06-a4b023c1b94f',
#  '008dcfbf-2994-4317-b5d1-e6a39c231154', '923faab6-937e-4957-ad01-1c08269a6b6e', '8b882024-4cf7-44cd-a5a8-a26ba069ead9',
#  '99032d87-4a63-4f10-ade3-137962ac1ad8', '605c88f9-8592-4fb1-ae84-1943d8db19eb', '52952ca5-a41a-4bef-91b8-b774e5b38935',
#  'a3f07f22-0b99-4c99-8e32-2c40c3ca1429', '7aa9a865-3f6c-419c-bd8c-3cc21e09c619', '8704d69c-bc9d-443f-9d9e-0cfdb5c281d2',
#  '13c13aa7-016c-4781-bd67-ab4581b5767e', '61943e6a-7756-43a2-b85b-ed6de0db0214', '28290576-e009-42b3-965a-cdeac5c8e28a'],
#  'embeddings': None, 'documents': ['삼성전자 사업 전망\n\n삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다.
#  여러 사업을 보유한 구조는 특정 시장의 부진을 다른 사업이 일부 보완할 수 있다는 장점이 있다.
#  다만 각 사업의 성장 요인과 위험이 다르므로, 회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.',
#  '반도체 부문에서 주목할 변화는 인공지능 데이터센터의 확산이다. 대규모 언어 모델을 학습하고 서비스하려면 높은 연산 성능뿐 아니라 데이터를 빠르게 주고받는 메모리가 필요하다.
#  이에 따라 고대역폭 메모리와 서버용 D램에 대한 관심이 커지고 있다.
#  삼성전자가 제품 성능과 생산 수율을 개선하고 주요 고객사의 품질 검증을 통과한다면, 인공지능 인프라 투자 확대를 매출 성장으로 연결할 가능성이 있다.',
# 
#  다만 각 영역에서 실제 매출과 이익이 얼마나 발생하는지는 시기별 자료로 확인해야 한다.',
#  '게임용 GPU, 전문시각화, 자동차 분야는 데이터센터 외의 사업 기반을 제공한다. 다만 인공지능 데이터센터 사업의 비중이 커질수록 실적은 대형 고객의 설비 투자 변화에 더 민감해질 수 있다.
#  중장기 전망을 평가할 때에는 인공지능 시장의 성장, 소프트웨어생태계의 지속성, 고객의 투자 수익성, 경쟁사의 대체 기술, 제품 공급 능력과 전력 비용을 함께 검토해야 한다.
#  이 문서는 RAG 실습을 위한 교육용 자료이며 투자 권유가 아니다.'],
#  'uris': None, 'included': 
# ['metadatas', 'documents'],
#  'data': None, 'metadatas': 
# [{'source': './_data/rag_data/samsung_outlook.txt'}, {'source': './_data/rag_data/samsung_outlook.txt'}, {'source': './_data/rag_data/samsung_outlook.txt'},
#  {'source': './_data/rag_data/samsung_outlook.txt'}, {'source': './_data/rag_data/samsung_outlook.txt'}, {'source': './_data/rag_data/samsung_outlook.txt'},
#  {'source': './_data/rag_data/samsung_outlook.txt'}, {'source': './_data/rag_data/samsung_outlook.txt'}, {'source': './_data/rag_data/samsung_outlook.txt'},
#  {'source': './_data/rag_data/nvidia_outlook.txt'}, {'source': './_data/rag_data/nvidia_outlook.txt'}, {'source': './_data/rag_data/nvidia_outlook.txt'},
#  {'source': './_data/rag_data/nvidia_outlook.txt'}, {'source': './_data/rag_data/nvidia_outlook.txt'}, {'source': './_data/rag_data/nvidia_outlook.txt'},
#  {'source': './_data/rag_data/nvidia_outlook.txt'}, {'source': './_data/rag_data/nvidia_outlook.txt'}, {'source': './_data/rag_data/nvidia_outlook.txt'}]}

print("============================================")
aaa = db.similarity_search("삼성전자 사업전망에 대해 알려줘.", k=2)   # 디폴트 4개

print(aaa)

# [Document(id='dcb3956b-044d-4ed2-8c41-b9320849297c', 
# metadata={'source': './_data/rag_data/samsung_outlook.txt'}, 
# page_content='삼성전자 사업 전망\n\n삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다.
#  여러 사업을 보유한 구조는 특정 시장의 부진을 다른 사업이 일부 보완할 수 있다는 장점이 있다.
#  다만 각 사업의 성장 요인과 위험이 다르므로, 회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.'), 
# Document(id='8b882024-4cf7-44cd-a5a8-a26ba069ead9', 
# metadata={'source': './_data/rag_data/samsung_outlook.txt'}, 
# page_content='삼성전자의 중장기 전망은 인공지능 메모리의 경쟁력, 
# 첨단 공정의 수율 개선, 파운드리 고객 확대, 
# 스마트폰의 제품 차별화에 달려 있다.
#  위험 요인으로는 반도체 가격 하락, 세계 경기 둔화, 공급망 차질, 환율 변동, 수출 규제와 경쟁 심화를 들 수 있다.
#  전망을 분석할 때에는 성장 산업에 참여하고 있다는 점과 그 기회가 실제 매출 및 이익으로 이어지는지를 함께 평가해야 한다.
#  이 문서는 RAG 실습을 위한 교육용 자료이며 투자 권유가 아니다.')]