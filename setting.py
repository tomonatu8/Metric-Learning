import ruamel.yaml

DEFAULT_DICT="""
common:
  random_seed: 0
  input_channel_num: 3
  output_class_num: 1 #変更
  dataset: cifar10
  num_supervised: 10 #変えてもいい。何枚既知とするか。100枚
moco: #Momentum Contrast
  batch_size: 100 #変更
  queue_size: 100000 #変更
train:
  num_batch_in_epoch: 100 #変更
  num_labeled_batch: 3 #変更
  patience: 20
  lr: 0.0001
  bw: 0.1 #バンド幅 カーネルの推定でラベル推定。
loss:
  sst: true #semi-super
  ce: false #cross entropy
  hoffer: false #論文の8
model:
  input_size: {height: 32, width: 32}
  feat_dim: 10 #変更
"""


CONFI_HAS_KEY=[("common","moco","train","loss","model"),("input_size","feat_dim")]


class Setting():
    def __init__(self,filepath=None):
        if filepath==None:
            self._setting_dict = ruamel.yaml.safe_load(DEFAULT_DICT) #上を読み込んでいるだけ
        else:
            try:
                with open(filepath,encoding='utf-8') as f:
                    self._setting_dict = ruamel.yaml.safe_load(f)
            except IOError as e:
                raise IOError("{0} [{1}]".format("error",e))
        if not (all(x in self._setting_dict for x in CONFI_HAS_KEY[0]) and all(x in self._setting_dict["model"] for x in CONFI_HAS_KEY[0])):
            raise KeyError("{0}".format("Error"))

    @property
    def common(self):
        return self._setting_dict["common"]
    @property
    def moco(self):
        return self._setting_dict["moco"]
    @property
    def train(self):
        return self._setting_dict["train"]
    @property
    def loss(self):
        return self._setting_dict["loss"]
    @property
    def model(self):
        return self._setting_dict["model"]
