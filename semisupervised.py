import tensorflow.keras
from tensorflow.keras.callbacks import Callback
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
#import pdb; pdb.set_trace()
from pdb import set_trace
from tensorflow.keras.datasets import cifar10

import logging

def load_cifar10():
    (x_train,y_train),(x_test,y_test)=cifar10.load_data()
    #>>> X_train.shape (50000, 32, 32, 3)
    input_shape=x_train.shape[1:] #(32, 32, 3)

    num_train=x_train.shape[0]
    num_test=x_test.shape[0]


    y_train=y_train.reshape((num_train,1)).astype('int32')
    y_test=y_test.reshape((num_test,1)).astype('int32')

    #normalize data
    x_train=x_train.astype('float32') / 255
    x_test=x_test.astype('float32') / 255

    return x_train, y_train, x_test, y_test


def semi_supervised_data(y_train,num_sp,num_class=10):
    y_train_unlabel=np.copy(y_train)
    for i in range(num_class):
        idx=np.where(y_train==i)[0]
        #idxはクラスがiの部分
        np.random.shuffle(idx)
        y_train_unlabel[idx[num_sp:]]=-1
        #クラスの同じもののうち、10個以外を全てクラス-1に変換した。
    return y_train_unlabel
