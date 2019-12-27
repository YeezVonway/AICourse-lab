from easydict import EasyDict as edict
import os

GLOBAL = edict()
GLOBAL.DEVICE = 'cuda'          # 高性能计算设备

## ..数据格式配置 ##

GLOBAL.DATA = edict()

## ....数据表示格式 ##

GLOBAL.DATA.SR = 22050              # 采样率
GLOBAL.DATA.FRAME_SIZE = 128      # 每帧的长度
GLOBAL.DATA.NUM_FRAMES = 1024       # 每个样本的帧的个数
GLOBAL.DATA.USE_MEL = False          # 使用梅尔频谱
GLOBAL.DATA.ARGUMENTATION = {    # 数据增强参数, None表示不使用数据增强
  'noiseStd': 1e-3,
  'wvScaleRg': 0.4,
  'tmScaleRg': 0.1,
  }
#GLOBAL.DATA.ARGUMENTATION = None

## ....频谱图配置 ##

GLOBAL.DATA.MEL = edict()
GLOBAL.DATA.MEL.FFT = 1024        # 快速傅里叶变换的采样数
GLOBAL.DATA.MEL.BIN = 128         # MEL滤波器个数
GLOBAL.DATA.MEL.SR = None         # 采样率

## ....类别 ##

GLOBAL.DATA.CLASSES = 'ACCORDINGLY' # 类别。包括名称和权重，决定于训练集。

## ..训练配置 ##

GLOBAL.TRAIN = edict()
GLOBAL.TRAIN.BATCH_SIZE = 256       # batch size
GLOBAL.TRAIN.NUM_WORKERS = 5      # 训练时，数据获取的进程数
GLOBAL.TRAIN.SHUFFLE = True       # 每次遍历数据集时，是否洗牌
GLOBAL.TRAIN.LR = 5e-5            # 学习率
GLOBAL.TRAIN.BETAS = (0.9, 0.99)  # Adam算法中的beta
GLOBAL.TRAIN.WEIGHT_DECAY = 0     # 权重的L2-loss系数
GLOBAL.TRAIN.SHOW_LOG = True      # 打印日志输出
GLOBAL.TRAIN.SHOW_PLOT = True     # 打印训练过程图表
GLOBAL.TRAIN.EPSILON = 1e-12      # 防止log爆炸的增值

## ..测试配置 ##

GLOBAL.TEST = edict()
GLOBAL.TEST.BATCH_SIZE = 256      # 测试时每批的大小
GLOBAL.TEST.NUM_WORKERS = 0       # 测试时，数据获取的进程数
GLOBAL.TEST.VERBOSE = True        # 为每个样本结果作输出

## 文件配置 ##

GLOBAL.FILE = edict()
GLOBAL.FILE.TRAIN_WAV = 'data\\train_wav'
GLOBAL.FILE.TEST_WAV = 'data\\test_wav'
GLOBAL.FILE.TRAIN_PKG = 'data\\trainset.pkg'  # 训练集打包文件
GLOBAL.FILE.TEST_PKG = 'data\\testset.pkg'    # 测试集打包文件
GLOBAL.FILE.MODELS = {   # key = 模型类型, value = 模型保存的路径
  'ReSENetWav' : '.\\models\\ReSE_Wav',
  # 'ReSENetMel' : '.\\models\\ReSE_Mel', # 算力不足，弃用
  'GoogLeNet' : '.\\models\\GoogLeNet'
  }
GLOBAL.FILE.EVAL_MODEL = 'ReSENetWav'  # 进行预测的模型

## 配置相关的工具函数 ##

def commit(cfg = GLOBAL):
  '''
  确认提交配置更改
  '''

  cfg.DATA.MEL.SR = cfg.DATA.SR
  if(cfg.DATA.USE_MEL):
    cfg.DATA.IN_CHANNELS = cfg.DATA.MEL.BIN
  else:
    cfg.DATA.IN_CHANNELS = cfg.DATA.FRAME_SIZE
  if(cfg.FILE.EVAL_MODEL not in cfg.FILE.MODELS):
    raise ValueError("用于预测的模型未定义")
  
  if cfg.DATA.CLASSES == 'ACCORDINGLY':
    get_classes_from_trainset(cfg)
  cfg.DATA.NUM_CLASS = len(cfg.DATA.CLASSES)

def get_classes_from_trainset(cfg=GLOBAL):
  '''
  从训练集文件目录更新cfg.DATA.CLASSES
  '''
  names = os.listdir(cfg.FILE.TRAIN_WAV)
  nums = [len(os.listdir(f"{cfg.FILE.TRAIN_WAV}\\{name}")) 
    for name in names]
  num_total = sum(nums)
  cfg.DATA.CLASSES = [(name, num_total/num) \
    for name, num in zip(names, nums)]

def addFileRoot(root='.\\', cfg=GLOBAL):
  '''
  为cfg配置的路径进行根转换
  '''

  if root[-1] not in ('\\', '/'):
    root += '\\'
  cfg.FILE.TRAIN_WAV = root + cfg.FILE.TRAIN_WAV
  cfg.FILE.TEST_WAV = root + cfg.FILE.TEST_WAV
  cfg.FILE.TRAIN_PKG = root + cfg.FILE.TRAIN_PKG
  cfg.FILE.TEST_PKG = root + cfg.FILE.TEST_PKG
  for key, path in cfg.FILE.MODELS.items():
    cfg.FILE.MODELS[key] =  root + path


commit()