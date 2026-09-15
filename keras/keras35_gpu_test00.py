'''
2026-09-15 (화)
'''

import tensorflow as tf
# print(tf.__version__)
# 2.9.3
gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]


if(gpus):
    print('GPU 있다 !~!')
else:
    print('GPU 없다 !~!')

# py311     에선 GPU 없다 !~! 출력
# tf29x-gpu 에선 GPU 있다 !~! 출력


