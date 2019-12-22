import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class GlobalAvgPool1d(nn.Module):
  '''
  1维“全局”平均值池化，即Squeeze操作层  \n
  ------
  输入: N * C * L 张量  \n
  输出: N * C 张量，池化的结果  \n
  + N: 样本数
  + C: 特征的通道数
  + L: 每个样本的长度
  '''

  def __init__(self):
    
    super().__init__()

  def forward(self, x):

    return torch.sum(x, dim = -1) / x.shape[-1]


class ReSEBlock1D(nn.Module):
  '''
  基于1维卷积的残差-SE网络块 \n
  ------
  输入: N * C * L 张量  \n
  输出: N * C * L 张量，池化的结果  \n
  + N: 样本数
  + C: 输入和输出特征的通道数
  + L_in: 每个样本输入和输出的长度
  ''' 
  
  def __init__(self, C, C_exc=None,
    ):
    '''
    + `C` : 输入每个特征点的通道数
    + `C_exc`: excitation中间层特征通道数，如果None，使用`C/16`
    '''
    
    super().__init__()
  
    if C_exc is None:
      C_exc = max(1, C//16)
  
    self.conv = nn.Conv1d(C, C, 3, 1, 1)
    self.sqz = GlobalAvgPool1d()
    self.exc_fc1 = nn.Linear(C, C_exc)
    self.exc_fc2 = nn.Linear(C_exc, C)

  def forward(self, X):

    U = self.conv(X)
    E = self.sqz(U)
    E = F.relu(self.exc_fc1(E))
    S = F.relu(self.exc_fc2(E))
    S = torch.unsqueeze(S, dim = -1)
    Y = U * S
    Y = F.relu(Y + X) #残差
    return Y

class ReSE2MpBnBlock1D(nn.Module):
  '''
  2层ReSE，加上maxpooling 和 bn
  ------
  输入: N * C_in * L_in 张量  \n
  输出: N * C_out * L_out 张量，池化的结果  \n
  + N: 样本数
  + C_in: 输入特征的通道数
  + C_out: 输出特征的通道数
  + L_in: 每个样本输入时的长度
  + L_out: 经过卷积后的样本长度，相当于L_in//2
  ''' 

  def __init__(self, C_in, C_out):

    super().__init__()

    self.rese1 = ReSEBlock1D(C_in)
    self.conv1x1 = nn.Conv1d(C_in, C_out, 1, 1, 0)
    self.rese2 = ReSEBlock1D(C_out)
    self.mxpl = nn.MaxPool1d(3, 2, 1)
    self.bn = nn.BatchNorm1d(C_out, 1e-12)

  def forward(self, X):

    X = self.rese1(X)
    X = self.conv1x1(X)
    X = self.rese2(X)
    X = self.mxpl(X)
    X = self.bn(X)

    return X