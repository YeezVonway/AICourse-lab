import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchData
import os

import CONFIG
import loss
import models
import data
from utils.log import Log

def get_trainData(cfg = CONFIG.GLOBAL):
  '''
  依据配置，获得数据集和loader  \n
  -------
  cfg: 配置，参考CONFIG.GLOBAL
  '''

  if cfg.DATA.USE_MEL:
    dataset = data.MelDataset(cfg.FILE.TRAIN_PKG, cfg)
  else:
    dataset = data.WavDataset(cfg.FILE.TRAIN_PKG, cfg)
  
  dataloader = dataset.get_loader(
    cfg.TRAIN.BATCH_SIZE, 
    cfg.TRAIN.NUM_WORKERS, 
    cfg.TRAIN.SHUFFLE)

  return dataset, dataloader

def train(cfg = CONFIG.GLOBAL, modelType = 'ReSENetWav', rnd = 1, logFile = None):
  '''
  训练网路模型  \n
  ------
  + `modelType`: 模型的类型名
  + `cfg`: 配置，参考CONFIG.GLOBAL
  + `rnd`: 遍历数据集的次数
  + `logFile` : 
    保存日志文件。如果已经存在该日志，则会在原基础上继续记录；否则会创建新日志。
    None表示不保存日志文件
  ------
  返回训练日志

  '''
  #准备训练
  dataset, dataloader = get_trainData(cfg)
  device = torch.device(cfg.DEVICE)
  model = models.get_model(modelType, cfg, True)
  model.train()
  Loss = loss.Loss(cfg).to(device = device)
  opt = torch.optim.Adam(
    model.parameters(),
    lr = cfg.TRAIN.LR,
    betas = cfg.TRAIN.BETAS,
    weight_decay = cfg.TRAIN.WEIGHT_DECAY,
    )

  #数据记录
  if logFile is not None and os.path.exists(logFile):
    log = Log.load(logFile)
  else:
    log = Log(correction = float, loss = float)

  for rnd_cnt in range(rnd):
    print(f'第{rnd_cnt}轮开始\n')

    for  X, Y in dataloader:
      X: torch.Tensor
      Y: torch.Tensor
      X = X.to(device = device)
      Y = Y.to(device = device)
      # 训练
      prediction = model(X)
      L = Loss(prediction, Y)
      L.backward()
      opt.step()
      # 日志记录
      with torch.no_grad():
        corr = torch.mean(prediction[range(Y.shape[0]),Y])
      log.record(correction = float(corr)*100, loss = float(L))

      if cfg.TRAIN.SHOW_LOG:
        log.print()

    print('正在保存--', end = '')
    models.save_model(model, cfg)
    if logFile is not None: log.save(logFile)
    if cfg.DEVICE != 'cpu':
      torch.cuda.empty_cache()
    print('完成')
  
  if cfg.TRAIN.SHOW_PLOT:
    log.make_plot()
  print(f'完成总计{rnd}轮训练')

if __name__ == "__main__":
  CONFIG.global_config("TRAIN.LR", 1e-6)
  train(rnd = 50, modelType = 'ReSENetWav', logFile = './trainlog.log')

    
