import torch
import torch.nn as nn
import numpy as np

class DebugError(Exception):

  def __init__(self, *args, **kargs):
    super(DebugError, self).__init__(*args, **kargs)

def nan_check_model(model: nn.Module):
  '''
  检查一个模型中的参数是否有nan
  '''

  for p in model.parameters():
    nan_check_tensor(p)
  
def nan_check_tensor(tens: torch.Tensor):
  '''
  检查一个tensor是否有nan
  '''
  checksum = float(torch.sum(tens))
  print(f'校验和 = {checksum}')
  if checksum != checksum:
    raise DebugError("发现nan")