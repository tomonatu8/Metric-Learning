import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from tqdm import tqdm

def get_semi_supervised_samples(x_train,y_ss_train,num_in_batch,num_class):
    batch_size=int(np.sum(num_in_batch))
    #num_in_batchだけに縮小
    x_shape = x_train.shape
    x = np.zeros((batch_size, x_shape[1], x_shape[2], x_shape[3]))
    #20
    y = np.zeros((batch_size,1))
    inds = np.zeros((batch_size,))

    idx = []
    for c in range(num_class):
        idx.append(np.where(y_ss_train==c)[0])
    idx_unlabel=np.where(y_ss_train==-1)[0]

    ptr=0
    for c in range(num_class):
        num_add = int(num_in_batch[c])
        np.random.shuffle(idx[c])
        pick = idx[c][:num_add]
        #同じクラスの中から、num_add、つまりbatchの分だけとり出す。
        x[ptr:ptr+num_add,:,:,:] = x_train[pick,:,:,:]
        #取り出したものをxに前から入れていく。
        y[ptr:ptr+num_add,:] = y_ss_train[pick,:]
        inds[ptr:ptr+num_add] = pick
        #indsはどこを取り出したか、覚えておく。
        ptr += num_add

    #今度はunlabelに対しても同じように選ぶ
    num_add = int(num_in_batch[num_class])
    np.random.shuffle(idx_unlabel)
    pick = idx_unlabel[:num_add]
    x[ptr:ptr+num_add,:,:,:] = x_train[pick,:,:,:]
    #取り出したものをxに前から入れていく。
    y[ptr:ptr+num_add,:] = y_ss_train[pick,:]
    inds[ptr:ptr+num_add] = pick
    #indsはどこを取り出したか、覚えておく。
    ptr += num_add

    return x, y, inds


def cos_sim(a, X):
    sim = K.dot(a, K.transpose(X))
    return K.transpose(sim)

def triplet_loss_from_cosine(P, N, triplet_margin):
    #配列の全要素と、0とで、maxをとっていると考えて良い。Noneはmaxならしい。
    return K.clip(N-P+triplet_margin,0,None)

#これむり
def get_semi_supervised_samples__2(x_train,y_ss_train,num_in_batch,num_class,model):
    batch_size=int(np.sum(num_in_batch))
    #num_in_batchだけに縮小
    x_shape = x_train.shape
    x = np.zeros((batch_size, x_shape[1], x_shape[2], x_shape[3]))
    y = np.zeros((batch_size,1))
    inds = np.zeros((batch_size,))

    idx = []
    for c in range(num_class):
        idx.append(np.where(y_ss_train==c)[0])
    idx_unlabel=np.where(y_ss_train==-1)[0]
    Triplet_Margin=1.0


    ptr=0
    use_num_list=np.zeros(num_class)
    i=0
    x_list=[]
    y_list=[]
    inds_list=[]
    for c in range(num_class):
        k=0
        pred_c = model(x_train[idx[c]],training=True)
        anchor_idx = np.random.choice(range(len(pred_c)))
        a=list(range(len(pred_c)))
        except_positive_idx = np.random.choice(a[:anchor_idx]+a[anchor_idx+1:])
        #print(np.array(pred_c[anchor_idx]))
        #print(np.array(pred_c[except_positive_idx]))
        #print(np.dot(np.array(pred_c[anchor_idx]), np.array(pred_c[except_positive_idx])))
        P_sim = np.dot(np.array(pred_c[anchor_idx]), np.array(pred_c[except_positive_idx]))
        for cc in range(num_class):
            if cc == c:
                continue
            else:
                pred_cc = model(x_train[idx[cc]],training=True)
                other_idx = np.random.choice(range(len(pred_cc)))
                N_sim = np.dot(np.array(pred_c[anchor_idx]), np.array(pred_cc[other_idx]))
                if P_sim - N_sim +Triplet_Margin > 0:
                    x_list.append(x_train[idx[c]][anchor_idx])
                    x_list.append(x_train[idx[c]][except_positive_idx])
                    x_list.append(x_train[idx[cc]][other_idx])
                    y_list.append(y_ss_train[idx[c]][anchor_idx])
                    y_list.append(y_ss_train[idx[c]][except_positive_idx])
                    y_list.append(y_ss_train[idx[cc]][other_idx])
                    inds_list.append(idx[c][anchor_idx])
                    inds_list.append(idx[c][except_positive_idx])
                    inds_list.append(idx[cc][other_idx])
                    i+=1
                    k+=1
                    #print(i)
        if i > 100:
            break

    print(len(x_list))
    print(x.shape)
    print(batch_size)
    for j in range(batch_size):
        #print(j)
        x[j]=x_list[j]
        y[j]=y_list[j]
        inds[j]=inds_list[j]
    """
    pick = idx[c][:num_add]
    #同じクラスの中から、num_add、つまりbatchの分だけとり出す。
    x[ptr:ptr+num_add,:,:,:] = x_train[pick,:,:,:]
    #取り出したものをxに前から入れていく。
    y[ptr:ptr+num_add,:] = y_ss_train[pick,:]
    inds[ptr:ptr+num_add] = pick
    #indsはどこを取り出したか、覚えておく。
    """
    num_add = int(num_in_batch[c]*num_class)
    ptr += num_add

    #今度はunlabelに対しても同じように選ぶ
    num_add = int(num_in_batch[num_class])
    np.random.shuffle(idx_unlabel)
    pick = idx_unlabel[:num_add]
    x[ptr:ptr+num_add,:,:,:] = x_train[pick,:,:,:]
    #取り出したものをxに前から入れていく。
    y[ptr:ptr+num_add,:] = y_ss_train[pick,:]
    inds[ptr:ptr+num_add] = pick
    #indsはどこを取り出したか、覚えておく。
    ptr += num_add

    print(x.shape)
    print(y.shape)
    print(inds.shape)

    return x, y, inds


def get_semi_supervised_samples__3(x_train,y_ss_train,num_in_batch,num_class,model):
    batch_size=int(np.sum(num_in_batch))
    #num_in_batchだけに縮小
    x_shape = x_train.shape
    x = np.zeros((batch_size, x_shape[1], x_shape[2], x_shape[3]))
    y = np.zeros((batch_size,1))
    inds = np.zeros((batch_size,))

    idx = []
    for c in range(num_class):
        idx.append(np.where(y_ss_train==c)[0])
    idx_unlabel=np.where(y_ss_train==-1)[0]
    Triplet_Margin=1.0


    ptr=0
    use_num_list=np.zeros(num_class)
    i=0
    x_list=[]
    y_list=[]
    inds_list=[]
    num_class_ran=list(range(num_class))
    np.random.shuffle(num_class_ran)
    for c in num_class_ran:
        k=0
        pred_c = model(x_train[idx[c]],training=True)
        anchor_idx = np.random.choice(range(len(pred_c)))
        a=list(range(len(pred_c)))
        except_positive_idx = np.random.choice(a[:anchor_idx]+a[anchor_idx+1:])
        #print(np.array(pred_c[anchor_idx]))
        #print(np.array(pred_c[except_positive_idx]))
        #print(np.dot(np.array(pred_c[anchor_idx]), np.array(pred_c[except_positive_idx])))
        #P_sim = np.dot(np.array(pred_c[anchor_idx]), np.array(pred_c[except_positive_idx]))
        num_class_ran_2=list(range(num_class))
        np.random.shuffle(num_class_ran_2)
        for cc in num_class_ran_2:
            if cc == c:
                continue
            elif c < cc:
                pred_cc = model(x_train[idx[cc]],training=True)
                other_idx = np.random.choice(range(len(pred_cc)))
                N_sim = np.dot(np.array(pred_c[anchor_idx]), np.array(pred_cc[other_idx]))
                if N_sim > 0:
                    #下手に大きいとだめ！
                    x_list.append(x_train[idx[c]][anchor_idx])
                    #x_list.append(x_train[idx[c]][except_positive_idx])
                    x_list.append(x_train[idx[cc]][other_idx])
                    y_list.append(y_ss_train[idx[c]][anchor_idx])
                    #y_list.append(y_ss_train[idx[c]][except_positive_idx])
                    y_list.append(y_ss_train[idx[cc]][other_idx])
                    inds_list.append(idx[c][anchor_idx])
                    #inds_list.append(idx[c][except_positive_idx])
                    inds_list.append(idx[cc][other_idx])
                    i+=1
                    k+=1
                    #print(i)
            if k > 1:
                break
        if i > 100:
            break

    print(len(x_list))
    print(x.shape)
    print(batch_size)
    for j in range(20):
        print(j)
        x[j]=x_list[j]
        y[j]=y_list[j]
        inds[j]=inds_list[j]
    """
    pick = idx[c][:num_add]
    #同じクラスの中から、num_add、つまりbatchの分だけとり出す。
    x[ptr:ptr+num_add,:,:,:] = x_train[pick,:,:,:]
    #取り出したものをxに前から入れていく。
    y[ptr:ptr+num_add,:] = y_ss_train[pick,:]
    inds[ptr:ptr+num_add] = pick
    #indsはどこを取り出したか、覚えておく。
    """
    num_add = int(num_in_batch[c]*num_class)
    ptr += num_add

    #今度はunlabelに対しても同じように選ぶ
    num_add = int(num_in_batch[num_class])
    np.random.shuffle(idx_unlabel)
    pick = idx_unlabel[:num_add]
    x[ptr:ptr+num_add,:,:,:] = x_train[pick,:,:,:]
    #取り出したものをxに前から入れていく。
    y[ptr:ptr+num_add,:] = y_ss_train[pick,:]
    inds[ptr:ptr+num_add] = pick
    #indsはどこを取り出したか、覚えておく。
    ptr += num_add

    print(x.shape)
    print(y.shape)
    print(inds.shape)

    return x, y, inds



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from dppy.finite_dpps import FiniteDPP

def plot_with_dpp(pred_c,add):
    rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
    X_projected = rp.fit_transform(pred_c)
    plt.figure(figsize=(10, 10), dpi=50)
    for p in X_projected:
        plt.scatter(p[0],p[1],color="blue")
    for a in add:
        plt.scatter(X_projected[a][0],X_projected[a][1],color="red")
    plt.show()

def select_with_dpp(pred_c,k):
    rng = np.random.RandomState(1)
    pred_c = np.array(pred_c)
    #pred_shape=(10,100)
    #100はout_putの出力
    A = pred_c.dot(pred_c.T)
    DPP = FiniteDPP('likelihood', **{'L': A})
    add = DPP.sample_exact_k_dpp(size=k, random_state=rng)
    #[7, 1, 5, 9]みたいな
    print(add)
    return add



def get_semi_supervised_samples__4(x_train,y_ss_train,num_in_batch,num_class,model):
    batch_size=int(np.sum(num_in_batch))
    #num_in_batchだけに縮小
    x_shape = x_train.shape
    x = np.zeros((batch_size, x_shape[1], x_shape[2], x_shape[3]))
    #20
    y = np.zeros((batch_size,1))
    inds = np.zeros((batch_size,))

    idx = []
    for c in range(num_class):
        idx.append(np.where(y_ss_train==c)[0])
    idx_unlabel=np.where(y_ss_train==-1)[0]

    ptr=0
    for c in range(num_class):
        pred_c = model(x_train[idx[c]],training=True)
        num_add = int(num_in_batch[c])
        add = select_with_dpp(pred_c,num_add)
        #np.random.shuffle(idx[c])
        pick=[]
        for a in add:
            pick.append(idx[c][a])
        #pick = idx[c][:num_add]
        #同じクラスの中から、num_add、つまりbatchの分だけとり出す。
        #plot_with_dpp(pred_c,add)
        x[ptr:ptr+num_add,:,:,:] = x_train[pick,:,:,:]
        #取り出したものをxに前から入れていく。
        y[ptr:ptr+num_add,:] = y_ss_train[pick,:]
        inds[ptr:ptr+num_add] = pick
        #indsはどこを取り出したか、覚えておく。
        ptr += num_add

    #今度はunlabelに対しても同じように選ぶ
    #pred_unlabel = model(x_train[idx_unlabel],training=True)
    #print("len(idx_unlabel)={}".format(len(idx_unlabel)))
    num_add = int(num_in_batch[num_class])
    #add = select_with_dpp(pred_unlabel,num_add)
    np.random.shuffle(idx_unlabel)
    #pick=[]
    #for a in add:
    #    pick.append(idx_unlabel[a])
    #plot_with_dpp(pred_unlabel,add)
    #np.random.shuffle(idx_unlabel)
    pick = idx_unlabel[:num_add]
    x[ptr:ptr+num_add,:,:,:] = x_train[pick,:,:,:]
    #取り出したものをxに前から入れていく。
    y[ptr:ptr+num_add,:] = y_ss_train[pick,:]
    inds[ptr:ptr+num_add] = pick
    #indsはどこを取り出したか、覚えておく。
    ptr += num_add

    return x, y, inds
