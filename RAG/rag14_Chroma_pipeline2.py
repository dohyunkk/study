'''
2026-10-07
13 카피
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
    temperature=0,
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
다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 컨텍스트 관련 정보가 없다면,
"주어진 정보로는 답변할 수 없습니다."라고 말씀해 주세요.

컨텍스트 : {context}
질문 : {input}
답변 : 
""")

# 체인 만들기
docu_chain = create_stuff_documents_chain(model, prompt)   # prompt | model  = 검색된 문서들을 LLM에게 넣어서 답변을 만들게 하는 체인
rag_chain = create_retrieval_chain(retriever, docu_chain)  # 검색 | docu_chain

# 체인 실행
query = "삼성전자의 창업자는 누구인가요?"
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
tags=['Chroma', 'OpenAIEmbeddings'] vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x00000117A05EEA50> search_kwargs={'k': 2}
검색된 관련 문서 수 : 2
첫번째 관련 문서 내용 미리보기 : 따라서 사용자가 "정부가 확보하려는 GPU 규모는 얼마인가?"라고 질문하면 질문도 임베딩 ...
=============================================================================
model의 답변 : 반도체를 볼 때는 **“무엇을 만드는 회사인지”와 “현재 업황의 어느 구간인지”**를 함께 살펴보는 게 중요합니다.

1. **수요가 어디서 오는가**
   - AI·데이터센터, 스마트폰, PC, 자동차, 산업용 장비 등 제품별 수요가 다릅니다.
   - 특히 AI 수요가 늘어도 모든 반도체가 똑같이 수혜를 받는 것은 아닙니다. GPU·가속기, HBM, 고속 네트워크·전력 관련 칩 등으로 수혜가 나뉩니다.

2. **메모리 업황과 재고**
   - DRAM·NAND는 가격과 재고 변화에 민감한 산업입니다.
   - 판매량뿐 아니라 **가격, 재고 수준, 가동률**을 같이 봐야 업황 회복이나 둔화를 판단하기 쉽습니다.

3. **기술 경쟁력**
   - 선단 공정의 성능만이 아니라 **수율, 생산 안정성, 고객사의 제품 채택**이 중요합니다.
   - 칩을 여러 개 결합하는 첨단 패키징과 HBM처럼, 공정 미세화 외의 기술도 경쟁력을 좌우합니다.

4. **투자와 공급 능력**
   - 공장 증설에는 큰 비용과 시간이 듭니다. 설비투자 계획이 수요보다 앞서면 공급 과잉이 생길 수 있습니다.
   - 발표된 생산능력보다 실제 양산 시점과 수율을 확인하는 편이 낫습니다.

5. **산업 내 위치**
   - 반도체 설계(팹리스), 위탁생산(파운드리), 메모리, 장비·소재, 후공정은 실적을 좌우하는 요인이 서로 다릅니다.
   - 예를 들어 장비·소재 업체는 고객사의 투자 계획과 수주 흐름을, 파운드리는 가동률과 고객 확보를 주로 살펴볼 수 있습니다.

6. **재무와 지정학적 위험**
   - 매출 성장뿐 아니라 영업이익률, 현금흐름, 부채, 고객 집중도를 확인하세요.
   - 수출 규제, 보조금, 관세, 특정 지역에 대한 생산 의존도도 사업에 영향을 줄 수 있습니다.

**투자 관점**이라면 “AI 수혜주” 같은 테마만 보기보다, 실제 매출과 이익으로 이어지는지, 현재 주가에 기대가 얼마나 반영됐는지를 함께 확인하는 것이 좋습니다. 원하시면 **삼성전자·SK하이닉스 같은 메모리 기업**, **파운드리**, **반도체 장비주** 중 한 분야를 골라 체크포인트를 더 구체적으로 정리해 드릴게요.
=============================================================================
=============================================================================
response :
{'input': '삼성전자의 창업자는 누구인가요?', 'context': [Document(id='e1a05fd4-a035-4271-a909-7b3826426f2e', metadata={'source': './_data/rag_data\\samsung_outlook.txt'}, page_content='삼성전자 사업 전망\n\n삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다. 여러 사업을 보유한 구조는 특정시장의 부진을 다른 사업이 일부 보완할 수 있다는 장점이 있다. 다만 각 사업의 성장 요인과 위험이 다르므로, 회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.'), Document(id='15b6f80e-fade-4c63-8ad9-608c13ebdda7', metadata={'source': './_data/rag_data\\samsung_outlook.txt'}, page_content='삼성전자의 중장기 전망은 인공지능 메모리의 경쟁력, 첨단 공정의 수율 개선, 파운드리 고객 확대, 스마트폰의 제품 차별화에 달려 있다. 위험 요인으로는 반도체 가격 하락, 세계 경기 둔화, 공급망 차질, 환율 변동, 수출 규제와 경쟁 심화를 들 수 있다. 전망을 분석할 때에는 성장 산업에 참여하고 있다는 점과 그 기회가 실제 매출 및 이익으로 이어지는지를 함께 평가해야 한다. 이 문서는 RAG 실습을 위한 교육용 자료이며 투자 권유가 아니다.')], 'answer': '주어진 정보로는 답변할 수 없습니다.'}
===================================================
response.keys() :
dict_keys(['input', 'context', 'answer'])
===================================================
response['context'][0].page_content :
삼성전자 사업 전망

삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다. 여러 사업을 보유한 구조는 특정 시장의 부진을 다른 사업이 일부 보완할수 있다는 장점이 있다. 다만 각 사업의 성장 요인과 위험이 다르므로, 회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.
===================================================
answer :
주어진 정보로는 답변할 수 없습니다.
"""