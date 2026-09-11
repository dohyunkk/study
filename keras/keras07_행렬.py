# 2026-08-31 월

import numpy as np

x1 = np.array([1,2,3])     #x1 = (3,)
print("x1 = ", x1.shape)

x2 = np.array([[1,2,3]])   #x2 = (1,3)
print("x2 = ", x2.shape)

x3 = np.array([[1,2],[3,4]]) #x3 = (2,2)
print("x3 = ", x3.shape)

x4 = np.array([[1,2],[3,4],[5,6,]])      #(3,2)
print("x4 = ", x4.shape)            #x4 = (3,2)

x5 = np.array([[[1,2],[3,4],[5,6,]]]) #     (1,3,2)
print("x5 = ", x5.shape)              #x5 = (1,3,2)

x6 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])    #x6 =  (2, 2, 2)
print("x6 = ", x6.shape)

x7 = np.array([[[[[1,2,3,4,5],[6,7,8,9,10]]]]])      #(1,1,1,2,5)
print("x7 = ", x7.shape)                        #x7 =  (1, 1, 1, 2, 5)

x8 = np.array([[[1,2,3]],[[4,5,6]]])                 #(2,1,3)
print("x8 = ", x8.shape)                        #x8 =  (2, 1, 3)

x9 = np.array([[[[1]]],[[[2]]]])                     #(2,1,1,1)
print("x9 = ", x9.shape)                        #x9 =  (2, 1, 1, 1)
