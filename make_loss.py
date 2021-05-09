import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from tqdm import tqdm

def cos_sim(a, X):
    sim = K.dot(a, K.transpose(X))
    return K.transpose(sim)

def triplet_loss_from_cosine(P, N, triplet_margin):
    #配列の全要素と、0とで、maxをとっていると考えて良い。Noneはmaxならしい。
    return K.clip(N-P+triplet_margin,0,None)



def loss_PU_cosine_triplet_1vA_CustomRate(num_in_batch,weight_PN_loss=0.5,Triplet_Margin=1.0,num_class=10):
    num_in_batch = num_in_batch.astype(np.int32)
    num_unlabel = num_in_batch[num_class]
    #[200]
    num_samples = num_in_batch[:num_class]
    #[2,2,2,2,2,...,2]

    #tf.functionをつけると、早い。set_traceは付けれない。
    @tf.function
    def loss_function(y_true, y_pred, class_rate):
        nDim = K.int_shape(y_pred)[-1]
        #普通にshapeとり出しているだけ。
        #問題は、y_predの次元が(200,100)とかになっている。
        #nDim=100
        y_pred = K.reshape(y_pred,(-1,nDim))
        #結局、(200,100)みたいな形にしている。
        y_true = K.flatten(y_true)

        class_mask = []
        class_pred = []
        for c in range(num_class):
            class_mask.append(K.equal(y_true,c))
            class_pred.append(tf.boolean_mask(y_pred, class_mask[c]))

        unlabel_mask = K.equal(y_true, -1)
        unlabel_pred = tf.boolean_mask(y_pred, unlabel_mask)

        loss = 0
        total_anchor = 0
        for c in range(num_class):
            print(c)
            #全てのクラスについて計算する。
            #for p in range(num_samples[c]):
            for p in range(1):
                total_anchor += 1

                #loss_PU
                anchor = class_pred[c][p:p+1,:]
                #クラスの中の最初の1こ目
                except_positive_pred = K.concatenate([class_pred[c][:p,:],class_pred[c][p+1:,:]],axis = 0)
                #二つのベクトルを合体させてるだけ
                P_sim = cos_sim(anchor, except_positive_pred)
                #これは、距離を計算しているだけ？？

                if num_unlabel > 0:
                    U_sim=cos_sim(anchor, unlabel_pred)

                    P_to_U = K.tile(P_sim, (1, num_unlabel))
                    U_to_P = K.tile(U_sim, (1, num_samples[c]-1))
                    U_to_P = K.transpose(U_to_P)

                    loss_PU = K.mean(triplet_loss_from_cosine(P_to_U,U_to_P,Triplet_Margin))
                    #E L_{tri}(a,p,u,phi) = max(d(a,p,phi)-d(a,u,phi)+margin , 0)
                    #今の場合、d(a,p,phi) = P_to_U
                    #d(a,u,phi) = U_to_P
                    #K.meanは全ての要素の平均。triplet_loss_from_cosine(P_to_U,U_to_P,Triplet_Margin)は配列。
                    loss_PU = K.abs(loss_PU - class_rate[c]*Triplet_Margin)
                    #class_rate=[0.1,0.1,...,0.1] 10個のうち均等なので。
                    #後で、lossに足す。

                #negatives
                other_pred = None
                num_other = 0
                for cc in range(num_class):
                    if cc == c:
                        continue
                        #これは上で考えた。
                    if other_pred is None:
                        other_pred = class_pred[cc]
                    else:
                        other_pred = K.concatenate([other_pred, class_pred[cc]], axis = 0)
                    num_other += num_samples[cc]

                #loss_N
                N_sim = cos_sim(anchor, other_pred)
                P_to_N = K.tile(P_sim, (1, num_other))
                N_to_P = K.tile(N_sim, (1, num_samples[c]-1))
                N_to_P = K.transpose(N_to_P)

                loss_PN = K.mean(triplet_loss_from_cosine(P_to_N, N_to_P, Triplet_Margin))

                if num_unlabel > 0:
                    U_to_N = K.tile(U_sim, (1, num_other))
                    N_to_U = K.tile(N_sim, (1, num_unlabel))
                    N_to_U = K.transpose(N_to_U)

                    loss_UN = K.mean(triplet_loss_from_cosine(U_to_N, N_to_U, Triplet_Margin))
                    loss_UN = K.abs(loss_UN - (1.0 - class_rate[c])*Triplet_Margin)

                if num_unlabel > 0:
                    loss += weight_PN_loss * loss_PN + (1.0 - weight_PN_loss) * (loss_PU + loss_UN)
                else:
                    loss += loss_PN

        return loss/total_anchor
    return loss_function


def loss_PU_cosine_triplet_1vA_CustomRate__2(num_in_batch,weight_PN_loss=0.5,Triplet_Margin=1.0,num_class=10):
    num_in_batch = num_in_batch.astype(np.int32)
    num_unlabel = num_in_batch[num_class]
    #[200]
    num_samples = num_in_batch[:num_class]
    #[2,2,2,2,2,...,2]

    #tf.functionをつけると、早い。set_traceは付けれない。
    @tf.function
    def loss_function(y_true, y_pred, class_rate):
        #print(y_pred)
        nDim = K.int_shape(y_pred)[-1]
        print(nDim)
        #普通にshapeとり出しているだけ。
        #問題は、y_predの次元が(200,100)とかになっている。
        #nDim=100
        y_pred = K.reshape(y_pred,(-1,nDim))
        #結局、(200,100)みたいな形にしている。
        y_true = K.flatten(y_true)

        class_mask = []
        class_pred = []
        for c in range(num_class):
            class_mask.append(K.equal(y_true,c))
            class_pred.append(tf.boolean_mask(y_pred, class_mask[c]))

        unlabel_mask = K.equal(y_true, -1)
        unlabel_pred = tf.boolean_mask(y_pred, unlabel_mask)

        loss = 0
        total_anchor = 0
        for c in range(num_class):
            #全てのクラスについて計算する。
            #for p in range(num_samples[c]):
            #print(class_pred[c][1:,:])
            print(class_pred[0])
            l=K.int_shape(class_pred[c])[0]
            print("l={}".format(l))
            for p in range(num_samples[c]-1):
                total_anchor += 1

                #loss_PU
                anchor = class_pred[c][p:p+1,:]
                #クラスの中の最初の1こ目
                except_positive_pred = K.concatenate([class_pred[c][:p,:],class_pred[c][p+1:,:]],axis = 0)
                #二つのベクトルを合体させてるだけ
                P_sim = cos_sim(anchor, except_positive_pred)
                print("P_sim={}".format(P_sim))
                #これは、positive同士で、内積をとっている。コサイン距離。

                #ここからは、教師ありの部分のloss計算
                other_pred = None
                num_other = 0
                for cc in range(num_class):
                    if cc == c:
                        continue
                        #これは上で考えた。
                    if other_pred is None:
                        other_pred = class_pred[cc]
                    else:
                        other_pred = K.concatenate([other_pred, class_pred[cc]], axis = 0)
                    num_other += num_samples[cc]

                #loss_N
                N_sim = cos_sim(anchor, other_pred)
                P_to_N = K.tile(P_sim, (1, num_other))
                N_to_P = K.tile(N_sim, (1, num_samples[c]-1))
                N_to_P = K.transpose(N_to_P)

                loss_supervised = K.mean(triplet_loss_from_cosine(P_to_N, N_to_P, Triplet_Margin))

                loss += loss_supervised



                #ここからは、教師なしの部分のloss計算
                if num_unlabel > 0:
                    U_sim = cos_sim(anchor, unlabel_pred)

                    sim_ap = K.mean(P_sim)
                    sim_au = K.mean(U_sim)
                    sim_an = K.mean(N_sim)

                    loss += K.abs(sim_au - class_rate[c] * sim_ap - (1.0 - class_rate[c]) * sim_an)

        return loss/total_anchor
    return loss_function


#サンプルとlossを一緒にしてしまう。
#まだサンプリングできてない
def loss_PU_cosine_triplet_1vA_CustomRate__3(num_in_batch,weight_PN_loss=0.5,Triplet_Margin=1.0,num_class=10):
    num_in_batch = num_in_batch.astype(np.int32)
    num_unlabel = num_in_batch[num_class]
    #[180]
    num_samples = num_in_batch[:num_class]
    #[2,2,2,2,2,...,2]

    #tf.functionをつけると、早い。set_traceは付けれない。
    @tf.function
    def loss_function(y_ss_train,pred,num_in_batch,num_class,class_rate,random_list):
        print("kokoko")
        nDim = K.int_shape(pred)[-1]
        #普通にshapeとり出しているだけ。
        #問題は、y_predの次元が(200,100)とかになっている。
        #nDim=100
        y_pred = K.reshape(pred,(-1,nDim))
        #結局、(200,100)みたいな形にしている。
        y_true = K.flatten(y_ss_train)

        class_mask = []
        class_pred = []
        for c in range(num_class):
            class_mask.append(K.equal(y_true,c))
            class_pred.append(tf.boolean_mask(y_pred, class_mask[c]))

        unlabel_mask = K.equal(y_true, -1)
        unlabel_pred = tf.boolean_mask(y_pred, unlabel_mask)

        loss = 0.0
        total_anchor = 0
        print("class_pred={}".format(class_pred))
        print("unlabel_pred={}".format(unlabel_pred))
        batch_size=int(K.sum(num_in_batch))
        print("kokokokoko")
        #=200とか。
        #num_in_batchだけに縮小
        #x_shape = x_train.shape
        #x = np.zeros((batch_size, x_shape[1], x_shape[2], x_shape[3]))
        #y = np.zeros((batch_size,1))
        #inds = np.zeros((batch_size,))
        """
        idx = []
        for c in range(num_class):
            idx.append(np.where(y_ss_train==c)[0])
        idx_unlabel = np.where(y_ss_train==-1)[0]
        Triplet_Margin = 1.0
        """

        ptr=0
        i=0
        x_list=[]
        y_list=[]
        inds_list=[]
        total_anchor = 0
        for c in range(num_class):
            #pred_c = model(x_train[idx[c]],training=True)
            sim_ap = 0.0
            sim_an = 0.0
            #sim_apやsim_anは平均をとるためのもの。
            l=K.int_shape(class_pred[c])[0]
            print("l={}".format(l))
            sample_size=0
            for j in range(2):
                for k in range(num_samples[c]-1):
                    total_anchor += 1
                    anchor = class_pred[c][k:k+1,:]
                    #a=K.random_uniform((1,1),0,num_samples[c]-1,dtype=tf.int32)
                    #print(a)
                    #positive_idx=int(a[0][0])
                    print("random_list={}".format(random_list))
                    except_positive = class_pred[c][random_list[0]:random_list[0]+1,:]
                    #一個選ぶ。
                    P_sim = K.dot(anchor, K.transpose(except_positive))
                    print(P_sim)
                    for cc in range(num_class):
                        if cc == c:
                            continue
                        else:
                            #pred_cc = model(x_train[idx[cc]],training=True)
                            #b=K.random_uniform((1,1),0,num_samples[c]-1,dtype=tf.int32)
                            #other_idx=int(b[0][0])
                            negative = class_pred[cc][random_list[1]:random_list[1]+1,:]
                            N_sim = K.dot(anchor, K.transpose(negative))
                            if P_sim - N_sim + Triplet_Margin > 0:
                                loss_supervised = P_sim - N_sim + Triplet_Margin
                                loss += loss_supervised
                                sim_ap += P_sim
                                sim_an += N_sim
                                sample_size += 1
                                #print(i)
            if num_unlabel > 0:
                #pred__unlabel = model(x_train[idx_unlabel],training=True)
                sim_au =0.0
                U_sim = cos_sim(anchor, unlabel_pred)
                sim_au = K.mean(U_sim)

                sim_ap = K.mean(sim_ap)
                sim_ap = K.mean(sim_an)

                loss += K.abs(sim_au - class_rate[c] * sim_ap - (1.0 - class_rate[c]) * sim_an)
        return loss/total_anchor
    return loss_function





def get_semi_supervised_samples(x_train,y_ss_train,num_in_batch,num_class):
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







class EarlyStoppingKDE(Callback):
    def __init__(self,x_train,y_train,x_val,y_val,logger,num_class=10,bw=0.1,patience=0,verbose=1,mode='max',metric='euclid',baseline=None,restore_best_weights=True):
        super(EarlyStoppingKDE,self).__init__()
        #Callback のクラスのmethodを呼び出すことができる。(?)
        #学習が収束するかどうかをカーネル推定で判断している。

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.logger = logger
        self.num_class = num_class
        self.bw = bw
        self.patience = patience
        self.verbose = verbose
        #modeって何？
        self.metric = metric
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights

        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        if mode == 'max':
            self.monitor_op = np.greater
        self.kde = []
        for c in range(self.num_class):
            if self.metric == 'euclid':
                self.kde.append(KernelDensity(bandwidth=self.bw))

    def on_train_begin(self, logs=None):
        de







class Triplet_Train():
    def __init__(self, log, num_class, batch_size):
        self._logger = log
        self.num_class = num_class
        #num_class=num_output=output_num_classで、10とか
        self.batch_size = batch_size
        #batch_sizeは200とか

    @tf.function
    def _train_step(self,x_train,y_train,class_rate):
        #このselfは、どのself???
        with tf.GradientTape() as tape:
            pred=self._model(x_train,training=True)
            #predはmodelのoutputなので、100次元とか
            #modelはfit内
            loss=self._loss_method(y_train,pred,class_rate)
            #このloss_methodもfit内
            scaled_loss=self._optimizer.get_scaled_loss(loss)

        scaled_grads=tape.gradient(scaled_loss,self._model.trainable_variables)
        #trainable_variablesとは？
        scaled_grads=self._optimizer._aggregate_gradients(zip(scaled_grads,self._model.trainable_variables))

        gradients=self._optimizer.get_unscaled_gradients(scaled_grads)
        gradients=self._optimizer._clip_gradients(gradients)
        self._optimizer.apply_gradients(zip(gradients,self._model.trainable_variables),experimental_aggregate_gradients=False)

        self.train_loss(loss)
        #ここのtrain_lossは下のfit内の。fit内で呼び出される。
        #上のoptimizerも、fit内で。

    def fit(self,model,x_train,y_ss_train,x_test,y_test,setting:dict,y_train):
        self._model=model
        #インスタンス
        self._setting=setting
        class_rate=np.zeros((self.num_class,),dtype=np.float32)+1
        class_rate=class_rate/self.num_class
        #[0.1,0.1,...,0.1]

        num_in_batch = np.zeros((self.num_class+1,))
        #1つ多い
        num_unlabel = self.batch_size
        #これはtriplet lossの
        for i in range(self.num_class):
            num_in_batch[i] = self._setting['num_labeled_batch']
            num_unlabel -= self._setting['num_labeled_batch']
        num_in_batch[self.num_class] = num_unlabel
        #num_in_batchにはクラス毎のサンプル数が入っている。最後にはunlabelの数。
        #+1されたところ。
        opt_Adam = tf.optimizers.Adam(self._setting['lr'],amsgrad=True)
        self._optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt_Adam,'dynamic')
        #混合精度を実現するもの。FP16とFP32を混ぜ合わせるらしい。後、scalingが必要ならしい(?)
        self._loss_method = loss_PU_cosine_triplet_1vA_CustomRate(num_in_batch,weight_PN_loss=0.5,Triplet_Margin=1.0,num_class=self.num_class)
        self.train_loss = tf.metrics.Mean(name='train_loss')
        #ここまでは基本的なtraining設定


        x_train_label=x_train[y_ss_train[:,0]>=0,:,:,:]
        y_train_label=y_train[y_ss_train[:,0]>=0,:]
        es = EarlyStoppingKDE(x_train_label,y_train_label,x_test,y_test,self._logger,self.num_class,bw=self._setting['bw'],patience=self._setting['patience'],verbose=1,mode='max')
        #学習が収束したときに途中で打ち切る必要がある。それをCallbackで操作する。
        es.set_model(self._model)
        es.on_train_begin(self._logger)
        es.on_epoch_end(-1,self._logger)

        #下で出力?
        bar_format='{n_fmt} [{elapsed}{rate_fmt}{postfix}]'

        epoch=0

        while True:
            #どうなったら止まるのか？
            pbar = tqdm(range(self._setting['num_batch_in_epoch']), leave = True, bar_format=bar_format)
            for e_cnt in pbar:

                x,y,inds = get_semi_supervised_samples(x_train,y_ss_train,num_in_batch,self.num_class)
                #x,y,inds = get_semi_supervised_samples__2(x_train,y_ss_train,num_in_batch,self.num_class,self._model,self._loss_method,class_rate)
                #こいつはランダムサンプリング！
                #これのサイズは、num_in_batchのもの。

                self._train_step(x,y,class_rate.astype(np.float32))

                pbar.set_postfix(loss = float(self.train_loss.result()))
                #tqdmのバーの後ろに情報を追加する。

            es.on_epoch_end(epoch,self._logger)
            epoch += 1

            self.train_loss.reset_states()
            #これは何

            if self._model.stop_training:
                break

        return self._model
