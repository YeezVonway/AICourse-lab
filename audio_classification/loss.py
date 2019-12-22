import torch
import torch.nn as nn
import torch.nn.functional as nnf

import audio_classification.CONFIG as CONFIG

class Loss(nn.Module):
  '''
  一个安全的交叉熵损失层，
  提供小epsilon避免log爆炸
  '''

  def __init__(self, cfg):
    '''
    依据配置获得交叉熵Loss层  \n
    ------
    + `cfg`: 配置， 可参考CONFIG.GLOBAL
    '''
    super(Loss, self).__init__()
    weight = torch.tensor([w for _,w in cfg.DATA.CLASSES])
    self.loss = nn.NLLLoss(weight = weight)
    self.eps = cfg.TRAIN.EPSILON

  def forward(self, X, Y):

    X = torch.log(X + self.eps)
    X = self.loss(X, Y)
    
    if float(X) != float(X):
      raise RuntimeError("损失为Nan")
    
    return X