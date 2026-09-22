'''
2026-09-21


'''
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt
path = 'c:/study/_data/image/'

img = load_img(path + 'me.png', target_size = (100, 100))

# print(img) 
# # <PIL.Image.Image image mode=RGB size=200x200 at 0x1A9180DD660>
# print(type(img)) # <class 'PIL.Image.Image'>

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
# print(arr)
# print(arr.shape) # (200, 200, 3)
# print(type(arr)) # <class 'numpy.ndarray'>

arr = np.expand_dims(arr, axis = 0)     # 차원 증가
# print(arr)
print(arr.shape)  # (1, 200, 200, 3), predict에 넣을 수 있는 데이터가 되었다.


np_path = './_data/save_img/'

np.save(np_path + 'keras48_me100.npy', arr=arr)
