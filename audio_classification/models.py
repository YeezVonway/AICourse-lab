import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

import utils.DEBUG as dbg

import CONFIG
from utils.ReSE import ReSEBlock1D
from utils.GoogleNet import Inception, BasicConv1d

class ReSENetWav(nn.Module):
  '''
  使用原本波形作为输入的的ReSE网络模型  \n
  -------
  输入: N * C * L 张量
  输出: N * NUM_CLASS 张量，表示样本在不同类别上的置信度
  + N: 样本数
  + C: 特征通道数（每个帧的长度）
  + L: 样本长度（帧数）
  + NUM_CLASS: 类别的个数
  '''

  def __init__(self, cfg = CONFIG.GLOBAL):
    '''
    + `cfg`: 配置， 可参照CONFIG.GLOBAL
    '''
    
    super().__init__()

    C: int = cfg.DATA.IN_CHANNELS
    L: int = cfg.DATA.NUM_FRAMES

    self.conv1, C = nn.Conv1d(C, 256, 3, 1, 1), 256
    self.maxpool1, L = nn.MaxPool1d(3, 2, 1), L//2
    self.rese1 = ReSEBlock1D(C, C//8)
    self.rese2 = ReSEBlock1D(C, C//8)
    self.bn1 = nn.BatchNorm1d(C, 1e-8)

    self.conv2, C = nn.Conv1d(C, 256, 3, 1, 1), 256
    self.maxpool2, L = nn.MaxPool1d(3, 2, 1), L//2
    self.rese3 = ReSEBlock1D(C, C//8)
    self.rese4 = ReSEBlock1D(C, C//8)
    self.bn2 = nn.BatchNorm1d(C, 1e-8)
    
    self.conv3, C = nn.Conv1d(C, 512, 3, 1, 1), 512
    self.maxpool3, L = nn.MaxPool1d(3, 2, 1), L//2
    self.rese5 = ReSEBlock1D(C, C//16)
    self.rese6 = ReSEBlock1D(C, C//16)
    self.bn3 = nn.BatchNorm1d(C, 1e-8)

    self.conv4, C = nn.Conv1d(C, 512, 3, 1, 1), 512
    self.maxpool4, L = nn.MaxPool1d(3, 2, 1), L//2
    self.rese7 = ReSEBlock1D(C, C//16)
    self.rese8 = ReSEBlock1D(C, C//16)
    self.bn4 = nn.BatchNorm1d(C, 1e-8)

    C = L * C

    self.fc_1, C = nn.Linear(C, 512), 512
    self.fc_2, C = nn.Linear(C, 512), 512
    self.fc_3 = nn.Linear(C, cfg.DATA.NUM_CLASS)
    self.sm = nn.Softmax(dim = 1)

  def forward(self, X):
    
    X = torch.relu(self.conv1(X))
    X = self.maxpool1(X)
    X = torch.relu(self.rese1(X))
    X = torch.relu(self.rese2(X))
    X = self.bn1(X)
    
    X = torch.relu(self.conv2(X))
    X = self.maxpool2(X)
    X = torch.relu(self.rese3(X))
    X = torch.relu(self.rese4(X))
    X = self.bn2(X)
    
    X = torch.relu(self.conv3(X))
    X = self.maxpool3(X)
    X = torch.relu(self.rese5(X))
    X = torch.relu(self.rese6(X))
    X = self.bn3(X)

    X = torch.relu(self.conv4(X))
    X = self.maxpool4(X)
    X = torch.relu(self.rese7(X))
    X = torch.relu(self.rese8(X))
    X = self.bn4(X)

    X = torch.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))

    X = torch.relu(self.fc_1(X))
    X = torch.relu(self.fc_2(X))
    X = self.fc_3(X)

    X = self.sm(X)

    return X

class ReSENetMel(nn.Module):
  '''
  使用Mel频谱作为输入的的ReSE网络模型  \n
  -------
  输入: N * C * L 张量
  输出: N * NUM_CLASS 张量，表示样本在不同类别上的置信度
  + N: 样本数
  + C: 特征通道数（mel滤波器个数）
  + L: 样本长度（每个帧的长度）
  + NUM_CLASS: 类别的个数
  '''

  def __init__(self, cfg = CONFIG.GLOBAL):
    '''
    + `cfg`: 配置， 可参照CONFIG.GLOBAL
    '''
    
    super().__init__()

    C: int = cfg.DATA.IN_CHANNELS
    L: int = cfg.DATA.NUM_FRAMES

    self.conv1, C = nn.Conv1d(C, 256, 3, 1, 1), 256
    self.maxpool1, L = nn.MaxPool1d(3, 2, 1), L//2
    self.rese1 = ReSEBlock1D(C, C//8)
    self.rese2 = ReSEBlock1D(C, C//8)
    self.bn1 = nn.BatchNorm1d(C, 1e-8)

    self.conv2, C = nn.Conv1d(C, 256, 3, 1, 1), 256
    self.maxpool2, L = nn.MaxPool1d(3, 2, 1), L//2
    self.rese3 = ReSEBlock1D(C, C//16)
    self.rese4 = ReSEBlock1D(C, C//16)
    self.bn2 = nn.BatchNorm1d(C, 1e-8)

    self.conv3, C = nn.Conv1d(C, 512, 3, 1, 1), 512
    self.maxpool3, L = nn.MaxPool1d(3, 2, 1), L//2
    self.rese5 = ReSEBlock1D(C, C//16)
    self.rese6 = ReSEBlock1D(C, C//16)
    self.bn3 = nn.BatchNorm1d(C, 1e-8)

    C = L * C

    self.fc_1, C = nn.Linear(C, 512), 512
    self.fc_2, C = nn.Linear(C, 512), 512
    self.fc_3 = nn.Linear(C, cfg.DATA.NUM_CLASS)
    self.sm = nn.Softmax(dim = 1)

  def forward(self, X):
    
    X = torch.relu(self.conv1(X))
    X = self.maxpool1(X)
    X = torch.relu(self.rese1(X))
    X = torch.relu(self.rese2(X))
    X = self.bn1(X)
    
    X = torch.relu(self.conv2(X))
    X = self.maxpool2(X)
    X = torch.relu(self.rese3(X))
    X = torch.relu(self.rese4(X))
    X = self.bn2(X)

    X = torch.relu(self.conv3(X))
    X = self.maxpool3(X)
    X = torch.relu(self.rese5(X))
    X = torch.relu(self.rese6(X))
    X = self.bn3(X)

    X = torch.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
    X = torch.relu(self.fc_1(X))
    X = torch.relu(self.fc_2(X))
    X = self.fc_3(X)

    X = self.sm(X)

    return X

class GoogLeNet(nn.Module):

    def __init__(self, cfg = CONFIG.GLOBAL):

        C_in = cfg.DATA.IN_CHANNELS
        C_out = cfg.DATA.NUM_CLASS
        super().__init__()

        self.pre_layers = BasicConv1d(C_in, 192, kernel_size=3, padding=1)
        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool1d(8, stride=1)
        self.linear = nn.Linear(1024 * 57, C_out)

        self.sm = nn.Softmax(dim = 1)

    def forward(self, x):

        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.sm(out)

        return out


def get_model(modelType, cfg = CONFIG.GLOBAL, create = False):
  '''
  获得指定类型的模型  \n
  ------
  + `modelType`: 模型的类型，可以取值：
      + `'ReSENetWav'`: 基于波形的ReSE网络
      + `'ReSENetMel'`: 基于Mel频谱的ReSE网络
  + `cfg`: 配置，参考CONFIG.GLOBAL
  + `create`: 如果为True，则会在读取保存的模型文件失败时，
    初始化参数并写到该文件。否则将会报错。
  ------
  返回指定的模型
  '''
  
  if modelType not in cfg.FILE.MODELS:
    raise ValueError('未定义的模型类型')
  else:
    model = globals()[modelType](cfg)
    path = cfg.FILE.MODELS[modelType]
  
  try:
    model.load_state_dict(torch.load(path))
  except:
    if create:
      for p in model.parameters():
        if len(p.shape) >= 2:
          torch.nn.init.xavier_normal_(p)
        else:
          torch.nn.init.normal_(
            p, 
            std= 1 / np.sqrt(p.shape[0])
            )
      torch.save(model.state_dict(), path)
      print(f"读取模型失败，已创建新模型于{path}")
    else:
      raise RuntimeError(f'无法于{path}读取模型')
  return model.to(device = torch.device(cfg.DEVICE))

def save_model(model: nn.Module, cfg = CONFIG.GLOBAL):
  '''
  获得指定类型的模型  \n
  ------
  + `model`: 被保存的模型
  + `cfg`: 配置，参考CONFIG.GLOBAL
  ------
  '''

  modelType = model.__class__.__name__
  if modelType not in cfg.FILE.MODELS:
    raise ValueError('未定义的模型类型')
  path = cfg.FILE.MODELS[modelType]
  torch.save(model.state_dict(), path)