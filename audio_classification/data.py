import numpy as np
import torch
import torch.utils.data as torchData
import pickle

import utils
import CONFIG

class BasicDataset(torchData.Dataset):
  '''
  数据集的基类，不直接产生Batch。  \n
  可以用__getitem__获得元组((filepath, wav), classIdx)
  '''
  def __init__(self, dataPkg: str, device, cfg = CONFIG.GLOBAL): 
    '''
    + dataPkg: 数据文件的路径，可用utils.pack打包得来。
    + `cfg`: 配置， 参考`CONFIG.GLOBAL`
    '''

    super(BasicDataset, self).__init__()
    self.cfg = cfg
    with open(dataPkg, 'rb') as fpkg:
      dataDic: dict = pickle.load(fpkg)

    self.data: list = []
    self.labels: list = []

    for clsIdx in range(len(cfg.DATA.CLASSES)):
      clsName, _ = cfg.DATA.CLASSES[clsIdx]
      if clsName not in dataDic:
        print(f"警告：类别{clsName}在数据包{dataPkg}中没有样本")
      else:
        data = dataDic[clsName]
        self.labels += [clsIdx for sample in data]
        self.data += data

  def __getitem__(self, index):

    return self.data[index], self.labels[index]

  def __len__(self):

    return len(self.data)

  def get_loader(self, batch_size, worker_num, shuffle):
    '''
    获取dataloader的接口
    '''
    raise NotImplementedError

class WavDataset(BasicDataset):
  '''
  音频帧原始波形数据的数据集  \n
  ------
  batch格式: (batch_size * C *L) 张量
  + batch_size: 批样本数
  + C: 特征的通道数，即帧长度
  + L: 每个样本的长度，即帧个数
  '''

  def __init__(self, dataPkg: str, cfg = CONFIG.GLOBAL):
    '''
    + dataPkg: 数据文件的路径，可用utils.pack打包得来。
    + `cfg`: 配置， 参考`CONFIG.GLOBAL`
    '''

    super(WavDataset, self).__init__(dataPkg, cfg)

  def __getitem__(self, index):
    
    (_, wav), clsIdx =  super().__getitem__(index)
    sample = utils.get_frames(wav, self.cfg.DATA)
    return sample, clsIdx

  @staticmethod
  def collate_fn(samples):
    x = torch.tensor(
      [wav for wav, clsIdx in samples], 
      dtype = torch.float32,
      )
    y = torch.tensor(
      [clsIdx for wav, clsIdx in samples], 
      dtype = torch.int64
      )
    return x, y

  def get_loader(self, batch_size, num_workers, shuffle):
    '''
    获取dataloader对象  \n
    -------
    + batch_size: 批的样本数
    + num_workers: 用于产生数据的进程数
    + shuffle: 是否随机乱序地遍历数据集
    '''
    return torchData.DataLoader(
      self,
      num_workers = num_workers,
      shuffle = shuffle,
      collate_fn = WavDataset.collate_fn,
      batch_size = batch_size
    )

class MelDataset(WavDataset):
  '''
  音频帧梅尔频谱数据的数据集  \n
  ------
  batch格式: (batch_size * C *L) 张量
  + batch_size: 批样本数
  + C: 特征的通道数，即梅尔滤波器个数
  + L: 每个样本的长度，即帧个数
  '''

  def __init__(self, dataPkg: str, cfg = CONFIG.GLOBAL):
    '''
    + dataPkg: 数据文件的路径，可用utils.pack打包得来。
    + `cfg`: 配置， 参考`CONFIG.GLOBAL`
    '''
    
    super(MelDataset, self).__init__(dataPkg, cfg)
    
    

  def __getitem__(self, index):

    sample, clsIdx = super().__getitem__(index)
    return utils.get_mel(sample, self.cfg.DATA.MEL), clsIdx

  @staticmethod
  def collate_fn(samples):
    x = torch.tensor(
      [arr for arr, clsIdx in samples], 
      dtype = torch.float32
      )
    y = torch.tensor(
      [clsIdx for arr, clsIdx in samples], 
      dtype = torch.int64
      )
    return x, y

  def get_loader(self, batch_size, num_workers, shuffle):
    '''
    获取dataloader对象  \n
    -------
    + batch_size: 批的样本数
    + num_workers: 用于产生数据的进程数
    + shuffle: 是否随机乱序地遍历数据集
    '''
    
    return torchData.DataLoader(
      self,
      num_workers = num_workers,
      shuffle = shuffle,
      collate_fn = MelDataset.collate_fn,
      batch_size = batch_size
      )

def pack_dataset(cfg = CONFIG.GLOBAL):
  '''
  将数据集打包 \n
  ------
  + cfg: 配置对象，参考CONFIG.GLOBAL
  '''

  utils.pack(cfg.FILE.TRAIN_WAV, cfg.FILE.TRAIN_PKG, cfg.DATA.SR)
  utils.pack(cfg.FILE.TEST_WAV, cfg.FILE.TEST_PKG, cfg.DATA.SR)
  print("数据集已打包")

