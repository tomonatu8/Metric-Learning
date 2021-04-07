import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K



#model作るやつ
def make_dense_net_simple(shape=(32,32,3),output_size=100,activate=None):
    input = tf.keras.layers.Input(shape=shape)
    dense_net = DenseNet201(include_top=False,weights='imagenet',input_tensor=input,pooling='avg')
    #output_sizeは出力先の次元の数(埋め込み先)
    if activate is None:
        #dense層とは、ただの全結合の層のこと。
        output = tf.keras.layers.Dense(output_size)(dense_net.output)
    elif activate == 'l2_norm':
        output = tf.keras.layers.Dense(output_size, activation=None)(dense_net.output)
        output = tf.keras.layers.Lambda(lambda x: K.l2_normalize(x,axis=1))(output)
    else:
        output = tf.keras.layers.Dense(output_size,activation=activate)(dense_net.output)

    model = tf.keras.Model(inputs=input,outputs=output)
    return model
