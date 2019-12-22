import numpy as np
import librosa
from easydict import EasyDict as edict
import numba
import pickle
import os

def pack(dataDir, dst, sr):
  '''
  把音频文件转换成numpy数组并打包到文件。  \n
  --------
  + dataDir: 数据文件的路径，需要按照如下格式摆放：
      > |dataDir  \n
      > |---|class_1  \n
      > |---|---|sample1.xxx \n
      > |---|---|sample2.xxx \n
      > |---|---|... \n
      > |---|class_2  \n
      > |---|...  \n
      > |---|class_C  \n
    + `dst`: 打包输出的文件
    + `sr`: 采样率
  '''

  dataDic = dict()

  clsNames = os.listdir(dataDir)
  for clsName in clsNames:
    sampleList = list()
    clsDir = f"{dataDir}\\{clsName}"
    fnames = os.listdir(clsDir)
    cnt = 0
    for fname in fnames:
      audioFile = f"{clsDir}\\{fname}"
      wav, _ =  librosa.load(audioFile, sr = sr)
      sampleList.append((audioFile, wav))
      cnt += 1
      print(f"解析{audioFile}于类别{clsName}，\
         该类别已完成{cnt}/{len(fnames)}")
    dataDic[clsName] = sampleList

  with open(dst, 'wb') as fdst:
    print('正在保存--', end='')
    pickle.dump(dataDic, fdst)
    print('完毕')


@numba.jit
def get_frames(arr: np.ndarray, fsize, fnum):
  '''
  把波形数组变成指定格式音频波形帧的数组。 
  自动截去过长的部分，不足的部分将自动用0补全  \n
  -------
  参数：
  + `arr`: 输入数组，为音频原波形, 长度为L
  + `fsize`: 帧的长度
  + `fnum`: 帧的个数
  -------
  返回值：`(FRAME_SIZE * NUM_FRAMES)数组`
  '''

  need_len = fsize * fnum
  arr_len = len(arr)

  if arr_len < need_len:
    arr = np.concatenate((
      arr, 
      np.zeros((need_len - arr_len,), dtype = np.float32),
    ))
  arr = np.reshape(arr[:need_len], (fnum, fsize))

  return arr.T

def get_mel(arr: np.ndarray, cfg: edict):
  '''
  把音频帧数组的每个帧变成mel频谱 \n
  ------
  参数：
  + `arr`：输入数组, 尺寸为(N * frame_size)， 代表N个音频帧 
  + `cfg`：mel的配置对象，可参考CONFIG.GLOBAL.DATA.MEL
      + `cfg.FFT`: FFT的采样数
      + `cfg.BIN`: mel滤波器个数
      + `cfg.SR`: 采样率
  ------
  返回值: (cfg.BIN * N)数组，代表N个帧的mel频谱
  '''

  fn_fft = cfg.FFT
  fn_mels = cfg.BIN
  fsr = cfg.SR
  arr_mel: list = []

  for frame in arr:
    frame = np.asfortranarray(frame)
    Y = librosa.core.stft(y = frame, n_fft = fn_fft)[:, 0]
    X = np.abs(Y)
    mel_fli = librosa.filters.mel(sr = fsr, n_fft = fn_fft, n_mels = fn_mels)
    mel_spectrum = np.dot(mel_fli, X)
    arr_mel.append(mel_spectrum)
    
  return np.array(arr_mel).T

@numba.jit
def data_argumentation(arr: np.array, noiseStd=0.02, wvScaleRg=0.1, tmScaleRg=0.04):
  '''
  数据扩充。 \n
  ------
  + arr: 原音频波形
  + noiseStd: 增添的噪音的标准差 与 平均幅值 之比
  + wvScaleRg: 音量缩放变化范围
  + tmScaleRg: 对时间缩放变化之范围
  '''
  

  noise = np.random.randn(*arr.shape) * noiseStd * np.mean(np.abs(arr))
  wvScale = 1 + (np.random.rand() * 2 - 1) * wvScaleRg
  tmScale = 1 + (np.random.rand() * 2 - 1) * tmScaleRg

  arr = arr + noise
  arr = arr * wvScale

  xp = np.arange(0, len(arr), 1)
  x = np.arange(0, len(arr), 1/tmScale)
  arr = np.interp(x, xp, arr)

  return arr


  