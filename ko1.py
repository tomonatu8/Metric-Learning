from setting import Setting
from make_net import make_dense_net_simple
import semisupervised
import make_sample
import numpy as np
from make_loss import Triplet_Train


from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

def plot_with_dpp(pred_c):
    rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
    X_projected = rp.fit_transform(pred_c)
    plt.figure(figsize=(4, 4), dpi=50)
    for p in X_projected:
        plt.scatter(p[0],p[1],color="blue")
    plt.show()

def select_with_dpp(pred_,k):
    pred_c = np.array(pred_c)
    A = pred_c.dot(pred_c.T)
    DPP = FiniteDPP('likelihood', **{'L': A})
    pick = DPP.sample_exact_k_dpp(size=k, random_state=rng)
    #[7, 1, 5, 9]みたいな
    return pick



if __name__ == '__main__':
    x_train, y_train, x_test, y_test =semisupervised.load_cifar10()

    input_height=32
    input_width=32
    num_ch=3
    feat_dim=100

    model=make_dense_net_simple(shape=(input_height,input_width,num_ch),output_size=feat_dim,activate='l2_norm')


    num_class=10
    num_labeled_batch=2
    batch_size=200

    num_in_batch = np.zeros((num_class+1,))
    #1つ多い
    num_unlabel = batch_size
    #これはtriplet lossの
    for i in range(num_class):
        num_in_batch[i] = num_labeled_batch
        num_unlabel -= num_labeled_batch

    num_in_batch[num_class] = num_unlabel

    num_sp=10

    y_ss_train=semisupervised.semi_supervised_data(y_train,num_sp,num_class)

    x,y,inds = make_sample.get_semi_supervised_samples(x_train,y_ss_train,num_in_batch,num_class)
    print("good0")

    #print("x={}".format(x))
    #print("y={}".format(y))
    #print("inds={}".format(inds))
    #print("good0")
    #x,y,inds = make_sample.get_semi_supervised_samples__2(x_train,y_ss_train,num_in_batch,num_class,model)
    #print("good")
    #print("x={}".format(x))
    #print("y={}".format(y))
    #print("inds={}".format(inds))
    #print("good")

    pred = model(x,training=True)
    print("good2")

    from make_loss import loss_PU_cosine_triplet_1vA_CustomRate
    from make_loss import loss_PU_cosine_triplet_1vA_CustomRate__2
    from make_loss import loss_PU_cosine_triplet_1vA_CustomRate__3
    loss_method = loss_PU_cosine_triplet_1vA_CustomRate(num_in_batch,weight_PN_loss=0.5,Triplet_Margin=1.0,num_class=num_class)
    class_rate=np.zeros((num_class,),dtype=np.float32)+1
    class_rate=class_rate/num_class


    print(y.shape)

    loss=loss_method(y,pred,class_rate)

    print("pred={}".format(pred))

    print(pred.shape)

    print("loss={}".format(loss))

    print(loss.shape)


    loss_method2 = loss_PU_cosine_triplet_1vA_CustomRate__2(num_in_batch,weight_PN_loss=0.5,Triplet_Margin=1.0,num_class=num_class)
    loss2=loss_method2(y,pred,class_rate)

    print("loss2={}".format(loss2))

    print(loss2.shape)

    #loss3はサンプルとlossを一緒に考える。イメージはlossを大きくすることを考える。

    loss_method3 = loss_PU_cosine_triplet_1vA_CustomRate__3(num_in_batch,weight_PN_loss=0.5,Triplet_Margin=1.0,num_class=num_class)
    #pred = model(x_train,training=True)
    random_list=list(range(int(num_in_batch[0])))
    np.random.shuffle(random_list)
    print("kkkkkkkkkkkk")
    print(random_list)
    loss3=loss_method3(y,pred,num_in_batch,num_class,class_rate,random_list)

    print("loss3={}".format(loss3))

    print(loss3.shape)
