'''
2026-09-28 (월)

def split_x(dataset, size):                  # 함수 정의      : dataset을 size만큼 잘라주는 함수split_x 만들기
    aaa = []                                 # 빈 리스트 생성 : 잘라낸 데이터를 담을 빈 리스트
    for i in range(len(dataset) - size + 1): # 반복           : 자를 수 있는 횟수만큼 반복
        subset = dataset[i : (i+size)]       # 데이터 자르기  : i번째부터 size개만큼 자르기
        aaa.append(subset)                   # 리스트에 추가  : 잘라낸 데이터를 aaa에 추가
    return np.array(aaa)                     # 결과 반환      : aaa를 넘파이 배열로 바꿔서 반환

'''

import numpy as np

a = np.array(range(1,11))  # [1,2,3,4,5,6,7,8,9,10]
size = 5                   # timestep 사이즈

# print(a.shape)           # (10,)

def split_x(dataset, size):                  # 함수 정의      : dataset을 size만큼 잘라주는 함수split_x 만들기
    aaa = []                                 # 빈 리스트 생성 : 잘라낸 데이터를 담을 빈 리스트
    for i in range(len(dataset) - size + 1): # 반복           : 자를 수 있는 횟수만큼 반복
        subset = dataset[i : (i+size)]       # 데이터 자르기  : i번째부터 size개만큼 자르기
        aaa.append(subset)                   # 리스트에 추가  : 잘라낸 데이터를 aaa에 추가
    return np.array(aaa)                     # 결과 반환      : aaa를 넘파이 배열로 바꿔서 반환

bbb = split_x(a, size)
print(bbb)
print(bbb.shape) # (6, 5)

