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
  输入: N * C_in * L_in 张量  \n
  输出: N * C_out * L_out 张量，池化的结果  \n
  + N: 样本数
  + C_in: 输入特征的通道数
  + C_out: 输出特征的通道数（等于C_in）
  + L_in: 每个样本输入时的长度
  + L_out: 经过卷积后的样本长度，相当于
    `(L_in + 2 * padding - kernel_size) / stride + 1`
  ''' 
  
  def __init__(self, C_in, C_exc = None,
    ):
    '''
    + `C_in` : 输入每个特征点的通道数
    + `C_exc`: excitation中间层特征通道数，如果None，使用`C_out/16`
    '''
    
    super().__init__()
    C_out = C_in
    if C_exc is None:
      C_exc = max(1, C_out//16)
  
    self.conv = nn.Conv1d(C_in, C_out, 3, 1, 1)
    self.sqz = GlobalAvgPool1d()
    self.exc_fc1 = nn.Linear(C_out, C_exc)
    self.exc_fc2 = nn.Linear(C_exc, C_out)

  def forward(self, X):

    U = self.conv(X)
    E = self.sqz(U)
    E = torch.relu(self.exc_fc1(E))
    S = torch.relu(self.exc_fc2(E))
    S = torch.unsqueeze(S, dim = -1)
    Y = U * S
    Y = torch.relu(Y + X) #残差
     
    return Y
