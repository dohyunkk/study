'''
2026-09-28 (월)

'''

import numpy as np

a = np.array([[1,2,3,4,5,6,7,8,9,10],        # 삼성전자 주가
              [9,8,7,6,5,4,3,2,1,0],         # 해당 날짜의 온도
              ]).T   
size = 5
print(a.shape)    # (10, 2)

# 칠판처럼 짤라보아요 !!
# split_x

def split_x(dataset, size):                  
    aaa = []                                 
    for i in range(len(dataset) - size + 1): 
        subset = dataset[i : (i+size)]       
        aaa.append(subset)                   
    return np.array(aaa)                     

bbb = split_x(a, size)
# print(bbb)
# # [[[ 1  9], [ 2  8], [ 3  7]],
# #  [[ 2  8], [ 3  7], [ 4  6]],
# #  [[ 3  7], [ 4  6], [ 5  5]],
# #  [[ 4  6], [ 5  5], [ 6  4]],
# #  [[ 5  5], [ 6  4], [ 7  3]],
# #  [[ 6  4], [ 7  3], [ 8  2]],
# #  [[ 7  3], [ 8  2], [ 9  1]],
# #  [[ 8  2], [ 9  1], [10  0]]]

# print(bbb.shape) # (8, 3, 2)

