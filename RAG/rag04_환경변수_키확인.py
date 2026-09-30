'''
2026-09-30

내 환경변수에 openai_api_key가 있는지 없는지, 어떤 정보가 있는지 확인.
'''

import os
key = os.getenv("OPENAI_API_KEY")   

if key is None:
    print("OPENAI_API_KEY 없다")
else:
    print("키 길이 :", len(key))
    print("키 확인 :", key[:8]+ "..."+key[-4:])

'''
환경변수에 키가 있을 때 : OPENAI_API_KEY 없다
환경변수에 키가 있을 때 : 
'''