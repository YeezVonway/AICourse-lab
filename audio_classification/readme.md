# 源码文件说明

+ **config.py**: 包含了数据、模型、训练和测试相关的配置信息。可以通过调整其中设定的数值来控制相关的特性。
+ **loss.py**: 计算损失
+ **models.py**: 几种不同类型的模型的源码
+ **train.py**: 训练模型相关
+ **test.py**: 测试模型相关
+ **data.py**: 包含若干数据集类，为模型提供训练、测试的样本
+ **utils**目录: 包含了辅助工具性的程序，可以被当作包。
+ **CONFIG.py**: 配置信息，调整模型和训练的具体参数。
+ **eval.py**: 对单个音频进行预测，可以开启预测服务
+ **result-reader.cpp**: 提供的读取输出结果为C++对象的接口。

# 环境支持

必须需要安装以下python环境来部署模型：
+ python 3.7
+ pytorch
+ numpy
+ librosa
+ easydict
+ numba
+ tkinter

为了支持更多音频编码格式，需要安装ffmpeg。  
可以使用conda来安装：`conda install ffmpeg -c conda-forge`

# 数据的准备

## 原始音频数据

data.py 并不直接接收音频文件作为输入（因为批量读取太慢），需要对原始音频进行打包。打包之前，数据的原始音频文件都需要被按照如下方式组织起来：

> |数据根目录  
> |--->|类别1目录  
> |----|--->|音频文件1  
> |----|--->|音频文件2  
> |----|--->|音频文件3  
> |----|--->|......  
> |--->|类别2目录  
> |--->|类别3目录  
> |--->|......

## 数据打包

数据必须被打包成文件才能用来训练。

对于任意的原始音频数据目录，只要是按照上述方式组织的，可以仿照以下示例代码来将之打包。

```python
from utils import pack
pack('f:\\data\\wav-files', #音频数据的根目录
  'f:\\data\\datapkg.pkg', #打包的目标文件
  sr = 11025 ) #采样率
```

如果你把原始音频数据放在由**CONFIG.py**指定的目录中，分别是：
+ `GLOBAL.FILE.TRAIN_WAV`
+ `GLOBAL.FILE.TEST_WAV`

那么可以通过调用`data.pack_dataset()`方法快速打包, sh输出文件路径为：
+ `GLOBAL.FILE.TRAIN_PKG` (这也是 train.py 直接读取的数据集打包文件)
+ `GLOBAL.FILE.TEST_PKG` (这也是 test.py 直接读取的数据集打包文件) 

示例代码为：

```python
from data import pack_dataset
pack_dataset()
```

## 标签类别

CONFIG.py 中的配置 `GLOBAL.DATA.CLASSES` 是一个列表，包含了所有类别，其中出现的顺序代表了标签的顺序。

该列表的每个元素是一个二元组 (类别名, 权重)。
在为训练或测试初始化数据集时，会提取`GLOBAL.FILE.XXX_PKG`指向的打包数据，并在中扫描上述列表当中的全部类别名。如果数据包中没有某个类别，控制台就会打印警告，因为这个类别将无法被学习或测试。
权重控制损失函数结果中各个类别的rescale的比例，如果不同类别的样本数量不平衡就需要调整权重。

有一种特殊情况： `GLOBAL.DATA.CLASSES = 'ACCORDINGLY'`,表示依据训练数据自动识别。这时程序将会到训练音频数据目录`GLOBAL.FILE.TRAIN_WAV`下提取目录名作为类别，并依据各个类别中的样本数量自动计算权重。

## 数据格式的配置

可以在 CONFIG.py 中的 GLOBAL.DATA 部分进行相关配置。

# 增加新的模型

为了比较测试，将会准备几种不同的模型。
+ `ReSENetWav` ：使用原波形作为输入的 ReSE-Net模型
+ `RESENetMEL` ：也是ReSE-NET，需要先把原波形变成梅尔频谱，才能作为输入。

未来可能尝试添加一些别的模型。

CONFIG.py 中以下配置和模型有关的：
+ `GLOBAL.DATA.USE_MEL`: 是否使用MEL频谱作为输入。这会决定使用哪一种模型。
+ `GLOBAL.FILE.MODELS`: 一个字典，包含一系列`(模型类型 : 模型保存路径)`的键值对。

如果需要添加新的模型，你需要：
1. 在 models.py 当中编写模型代码。该模型必须满足以下条件：
    + 构造函数以配置字典为参数，比如 CONFIG.py 中的 `GLOBAL` 配置字典。比形如:
      ```python
      class MyNet(nn.Module):
        def __init__(self, cfg):
          ...
      ```
    + 能适合所给配置字典作为参数的输入数据。以`cfg`为所给配置参数，模型输入张量的尺寸为：`cfg.TRAIN.BATCH_SIZE, cfg.DATA.IN_CHANNELS, cfg.DATA.NUM_FRAMES`
    + 正确地重写`forward`方法，输出张量的尺寸为`cfg.TRAIN.BATCH_SIZE * cfg.DATA.NUM_CLASS`，表示每个样本在每个类别上的softmax得分。
2. 在配置 CONFIG.py 中注册该模型。你需要在 `GLOBAL.FILE.MODELS` 增添相应的条目，并确保所给的模型类型名等同于你在 models.py 中定义的类名。

# 训练模型

train.py 当中的train函数提供训练功能。示例如下：

```python
import train
train.train(
  'ReSENetWav', #训练ReSENetWav模型
  rnd = 10, #遍历训练集10次
  logFile = '.\\log')#保存日志到'.\log'
```

训练使用数据集由配置`GLOBAL.FILE.TRAIN_PKG`指定，务必确保该数据包存在，其中的类别名称（解包后可见）和`GLOBAL.DATA.CLASSES`中的类别一致，且其中的波形数据使用的采样率和配置是一致的。

模型保存路径由配置 `GLOBAL.FILE`的对应项指定。再训练开始前，程序会先读取保存的模型来继续上一次训练。但如果找不到保存的模型文件，或者保存的模型和要使用的模型无法匹配（比如更改了配置，使得模型发生了变化），那么程序将自动初始化一个新的模型并保存（会覆盖原本保存的模型）。因此，如果你可能用到原来的模型，记得备份起来。

训练过程将会一轮一轮地遍历训练集，并会再每轮结束时保存模型和日志。如果看到日志输出“正在保存”，那么在完成之前切忌按下强制结束键，否则模型可能就毁了。

如果训练出了bug，产生的loss是nan，程序会报错并终止，这将会保护模型参数被nan玷污。但还是记得多给模型备份！


## 相关配置

CONFIG.py 中的 `GLOBAL.TRAIN` 是和训练有关的，常见的参数有：
+ `GLOBAL.TRAIN.BATCH_SIZE`： batch size
+ `GLOBAL.TRAIN.NUM_WORKERS`: 训练时，用来喂给模型数据的进程数
+ `GLOBAL.TRAIN.SHUFFLE`: 每次遍历数据集时，是否对数据进行洗牌
+ `GLOBAL.TRAIN.LR`: 学习率
+ `GLOBAL.TRAIN.SHOW_LOG`: 是否打印日志输出
+ `GLOBAL.TRAIN.SHOW_PLOT`: 完成后，是否打印训练过程图表

# 测试模型

test.py 当中的test函数提供测试功能。示例如下：
```python
import test
test.test('ReSENetWav')
```
测试结果将在控制台被打印输出。

# 部署模型

eval.py 提供了使用模型进行部署的相关接口。

## 选择训练好的模型

CONFIG.py 中可以设置选择哪一个模型来进行部署。需要设置的条目是`GLOBAL.FILE.EVAL_MODEL`。

你必须确保这个条目的值是`GLOBAL.FILE.MODELS`当中的某一个键值。

## 启动服务

调用Python解释器运行 eval.py 。
```
>> python eval.py
```
这将会开启一个窗口，你可以从中设置**输入音频目录**和**输出结果目录**，并点击按钮启动或终止服务线程。你能够在控制台中查看服务的启动情况。

当服务运行时，所有往**输入音频目录**添加的音频文件都会被读取并分类，结果将会输出到**输出结果目录**，输出文件名只对应地更改了后缀名。与此同时，已经分类过的音频将会被移动到**输入音频目录**下的**evaled子目录**中。

## 输出数据格式

每一个输入，都会对应得到一个输出文件，只更改后缀名。假设输入是 audio.mp3，那么输出将是 audio.evl。

一个evl文件共包含两行：
+ 第一行是一个整数，即最大可能的标签类别的序号。
+ 第二行包括若干个浮点数，依次为原音频的各种类别为可能概率。

例如以下是一个evl文件，其中最大可能类别的序号是0，各种类别的可能性约为0.982, 0.007, 5.98e-08, 0.011。
> 0  
> 0.9816079139709473 0.007378084119409323 5.980089667900756e-08 0.011013897135853767 

# 工作目录问题

如果工作目录是audio_classification（也就是本说明文档所在的目录），那么所有程序能够正常工作。这也是使用此程序的建议方式。

但如果要在外部目录执行代码来使用本目录中的内容，就可能找不到包和文件。

以下是几种解决办法：
+ 在代码中使用`sys.path.append(本目录)`，把本目录添加到环境。
+ 在代码中使用`os.chdir()`来切换工作目录到本目录。

随后就能正常import本目录中的python文件。

另外，建议在配置`GLOBAL.FILE`当中使用绝对路径而不是相对路径。