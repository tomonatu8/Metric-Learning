import tensorflow as tf
import numpy as np
import random
import os, sys


import semisupervised
from setting import Setting
from make_net import make_dense_net_simple
from make_loss import Triplet_Train


def sst_train(setting_file,log_file):
    #入力はyaml, settingはyamlをdictに

    try:
        setting=Setting(setting_file)
    except IOError as e:
        print(e)

    seed=setting.common['random_seed']
    #乱数
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    #logとってるだけ
    import logging
    logger=tf.get_logger()
    logger.setLevel(logging.CRITICAL)
    logger=logger.get_logger(log_file)

    #パラメータをsettingからとってくる。
    num_ch=setting.common['input_channel_num']
    num_output=setting.common['output_class_num']
    dataset=setting.common['dataset']
    num_sp=setting.common['num_supervised'] #ラベルのついているやつ

    input_height=setting.model['input_size']['height']
    input_width=setting.model['input_size']['width']
    feat_dim=setting.model['feat_dim']

    #mocoの中にある
    batch_size=setting.moco['batch_size']
    queue_size=setting.moco['queue_size']

    if dataset=="cifar10":
        x_train, y_train, x_test, y_test =semisupervised.load_cifar10()
        num_class=10
    else:
        print("error")
        return

    #num_spの分以外を消す。消されたところのクラスは-1
    y_ss_train=semisupervised.semi_supervised_data(y_train,num_sp,num_class)

    model=make_dense_net_simple(shape=(input_height,input_width,num_ch),output_size=feat_dim,activate='l2_norm')
    #32,32,3  100  
    #tranするクラスを考える。
    train=Triplet_Train(logger,num_output,batch_size)
    result=train.fit(model,x_train,y_ss_train,x_test,y_test,setting.train,y_train)

    model.save_weights('weight/weights_{}.h5'.format(dataset))



if __name__ == '__main__':
    args=sys.argv
    setting_file=arg[1]
    log_file=arg[2]

    #ここではtrainして、その重みを保存するという工程。
    sst_train(setting_file,log_file)
