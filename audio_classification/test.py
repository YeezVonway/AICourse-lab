import torch

import audio_classification.CONFIG as CONFIG
import audio_classification.models as models
import audio_classification.data as data
from audio_classification.utils.log import Log


def get_testData(cfg=CONFIG.GLOBAL):
  '''
  依据配置，获得数据集和loader  \n
  -------
  cfg: 配置，参考CONFIG.GLOBAL
  '''

  if cfg.DATA.USE_MEL:
    dataset = data.MelDataset(cfg.FILE.TEST_PKG, cfg)
  else:
    dataset = data.WavDataset(cfg.FILE.TEST_PKG, cfg)
  
  dataloader = dataset.get_loader(
    cfg.TEST.BATCH_SIZE, 
    cfg.TEST.NUM_WORKERS, 
    False)

  return dataset, dataloader

def test(modelType = 'ReSENetWav', cfg = CONFIG.GLOBAL):
  '''
  训练网路模型  \n
  ------
  + `modelType`: 模型的类型名
  + `cfg`: 配置，参考CONFIG.GLOBAL
  ------
  返回训练日志

  '''
  #准备测试
  dataset, dataloader = get_testData(cfg)
  device = torch.device(cfg.DEVICE)
  model = models.get_model(modelType, cfg, False)
  model.eval()

  #数据记录
  log = Log(num_correct = int, num_batch = int, mean_score = float)

  with torch.no_grad():
    for  X, Y in dataloader:
      X: torch.Tensor
      Y: torch.Tensor
      X = X.to(device = device)
      Y = Y.to(device = device)

      prediction = model(X)
      num_batch = X.shape[0]

      num_corr = torch.sum(
        (Y == torch.argmax(prediction, dim = -1)
        ).float())
      mean_score = torch.mean(prediction[range(len(Y)), Y])
      log.record(
        num_correct = int(num_corr), 
        num_batch = int(num_batch), 
        mean_score = float(mean_score))
      if cfg.TEST.VERBOSE:
        log.print()
      
    torch.cuda.empty_cache()
    total_correct = sum(log.itemValues['num_correct'])
    total_num = sum(log.itemValues['num_batch'])
    total_score = sum([n * s for n,s in zip(
      log.itemValues['num_batch'], 
      log.itemValues['mean_score']
      )])
    print(f"正确总数：{total_correct}/{total_num}，\
      正确率：{total_correct/total_num * 100}%, \
      正确类平均分: {total_score/total_num * 100}%")
  