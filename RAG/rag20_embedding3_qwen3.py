'''
20-2
'''


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip() # strip() 공백이나 줄바꿈을 무시 (오탈자로 인한 에러 방지용)
base_url = 'https://monogpt.kr/api/monorouter/v1'

prompt = '삼성전자의 창업주는 누구인가요?'          # PromptTemplate 이 아니라 그냥 문자열 -> 이 문장 자체를 벡터로 바꿀 대상

# from langchain_openai import OpenAIEmbeddings      # 문장 -> 벡터로 바꿔 주는 OpenAI 임베딩 모델 (ChatOpenAI 대신 사용)
# embeddings = OpenAIEmbeddings(
#     # model = 'text-embedding-3-small',            # 작은 모델 -> 1536 차원 (빠르고 저렴)
#     model = 'text-embedding-3-large',              # 큰 모델   -> 3072 차원 (문장 의미를 더 세밀하게 표현, 더 비쌈)
#     api_key=api_key,
#     base_url=base_url,                             # ChatOpenAI 와 마찬가지로 사제 키를 쓸 때는 key 와 URL 을 함께 넣어야 함
# )

# ================================================================
# pip install langchain-huggingface
# pip install sentence-transformers
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import sentence_transformers

embeddings = HuggingFaceEmbeddings(
    model_name = "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs ={
        "device" : "cpu",
        "local_files_only" : True
    } 
)

vector = embeddings.embed_query(prompt)        
                                                  
print(vector)                                      
print('==============================')
print('임베딩 벡터의 차원: ', len(vector)) 

# 임베딩 벡터는 한 개, 벡터 안에 차원이 1024개
