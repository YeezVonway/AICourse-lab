from easydict import EasyDict as edict
import os

GLOBAL = edict()
GLOBAL.DEVICE = 'cuda'          #高性能计算设备

## ..数据格式配置 ##

GLOBAL.DATA = edict()

## ....数据表示格式 ##

GLOBAL.DATA.SR = 11050              # 采样率
GLOBAL.DATA.FRAME_SIZE = 256       # 每帧的长度
GLOBAL.DATA.NUM_FRAMES = 256       # 每个样本的帧的个数
GLOBAL.DATA.USE_MEL = False          # 使用梅尔频谱
GLOBAL.DATA.IN_CHANNELS = None      # 每个样本特征点的通道数（由其它参数确定）

## ....频谱图配置 ##

GLOBAL.DATA.MEL = edict()
GLOBAL.DATA.MEL.FFT = 1024        # 快速傅里叶变换的采样数
GLOBAL.DATA.MEL.BIN = 128         # MEL滤波器个数
GLOBAL.DATA.MEL.SR = None         # 采样率

## ....类别 ##

GLOBAL.DATA.CLASSES = 'ACCORDINGLY' # 类别。包括名称和权重，决定于训练集。

GLOBAL.DATA.NUM_CLASS = None      # 类数，由CLASSES决定

## ..训练配置 ##

GLOBAL.TRAIN = edict()
GLOBAL.TRAIN.BATCH_SIZE = 256       # batch size
GLOBAL.TRAIN.NUM_WORKERS = 3      # 训练时，数据获取的进程数
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
GLOBAL.TEST.NUM_WORKERS = 3       # 测试时，数据获取的进程数
GLOBAL.TEST.VERBOSE = True        # 为每个样本结果作输出

## 文件配置 ##

GLOBAL.FILE = edict()
GLOBAL.FILE.TRAIN_WAV = '.\\data\\train_wav'
GLOBAL.FILE.TEST_WAV = '.\\data\\test_wav'
GLOBAL.FILE.TRAIN_PKG = '.\\data\\trainset.pkg'  # 训练集打包文件
GLOBAL.FILE.TEST_PKG = '.\\data\\testset.pkg'    # 测试集打包文件
GLOBAL.FILE.MODELS = {   # key = 模型类型, value = 模型保存的路径
  'ReSENetWav' : '.\\models\\ReSE_Wav',
  'ReSENetMel' : '.\\models\\ReSE_Mel'
  }

GLOBAL.FILE.RESE_MEL = '.\\models\\ReSE_Mel'  # 保存ReSEWav模型的路径

## 配置相关的工具函数 ##

def update():
  '''
  更新一些由其它参数决定的数值
  '''

  GLOBAL.DATA.NUM_CLASS = len(GLOBAL.DATA.CLASSES)
  GLOBAL.DATA.MEL.SR = GLOBAL.DATA.SR
  if(GLOBAL.DATA.USE_MEL):
    GLOBAL.DATA.IN_CHANNELS = GLOBAL.DATA.MEL.BIN
  else:
    GLOBAL.DATA.IN_CHANNELS = GLOBAL.DATA.FRAME_SIZE

def get_classes_from_trainset():
  '''
  从训练集文件目录更新GLOBAL.DATA.CLASSES
  '''
  names = os.listdir(GLOBAL.FILE.TRAIN_WAV)
  nums = [len(os.listdir(f"{GLOBAL.FILE.TRAIN_WAV}\\{name}")) 
    for name in names]
  num_total = sum(nums)
  GLOBAL.DATA.CLASSES = [(name, num/num_total) 
    for name, num in zip(names, nums)]

def global_config(name: str, value):
  '''
  更改GLOBAL参数配置\n
  ------
  + `name`: 参数名称
  + `value`：数值
  ------
  示例：
  
  `config('TRAIN.BATCH_SIZE', 128)`
  '''

  exec('GLOBAL.' + name +' = '+ str(value))
  update()

## 载入配置例程 ##

if GLOBAL.DATA.CLASSES == 'ACCORDINGLY':
  get_classes_from_trainset()

update()