import matplotlib.pyplot as plt
import pickle

class Log():
  '''
  训练和测试的日志输出工具。能够添加记录、保存到文件、打印图表。
  '''
  def __init__(self, **items):
    '''
    参数给出记录的条目和类型  \n
    -----
    示例：
    ```
    > log = Log(loss = float, correction = float)
    > log.record(loss = 1., correction = 2.)
    ```
    '''
    items: dict
    for k, v in items.items():
      if type(v) is not type:
        raise ValueError("必须指定类型")

    self.itemTypes = items
    self.itemValues = dict([(k,[]) for k in items])
    self.length = 0
    self.latest = None

  def __getitem__(self, index):
    
    if index >= self.length: raise IndexError
    return dict([ (k, v[index]) 
      for k,v in self.itemValues.items()])

  def __len__(self):

    return self.length

  def record(self, **kargs):
    '''
    添加一次记录。要求参数中恰好包括所有条目，且类型一致。\\n
    ------
    示例：
    ```
    > log = Log(dog = int, cat = float)
    > log.record(dog = 2, cat = 3.)
    ```
    '''
    kargs: dict

    for k, v in kargs.items():
      if k not in self.itemTypes:
        raise KeyError("无效的记录项")
      if type(v) is not self.itemTypes[k]:
        raise TypeError(
          f"类型不符。需要{self.itemsTypes[k]}, 所给的是{type(v)}"
          )
      
    for k, valueList in self.itemValues.items():
      valueList: list
      if k not in kargs:
        raise KeyError(f"项'{k}'未得到记录")
      valueList.append(kargs[k])

    self.length += 1
    self.latest = kargs

    
  def make_plot(self, items = None):
    '''
    绘出图表。\n
    ------
    + items: 希望打印图表的条目列表。None表示全部打印。
    '''
    if items is None:
      for k, v in self.itemValues.items():
        plt.figure()
        plt.plot(v)
        plt.title(k)
    else:
      for k in items:
        v = self.itemValues[v]
        plt.figure()
        plt.plot(v)
        plt.title(k)

    plt.show()

  def print(self, index = -1):
    '''
    打印从索引某一次记录
    '''
    if index == -1: index = self.length - 1
    print(f'#{index}: ')
    for k, v in self.__getitem__(index).items():
      print(f'\t{k} = {v}')

  def save(self, path):
    '''
    保存自身到指定路径。
    ''' 

    with open(path, 'wb') as f:
      pickle.dump(self, f)

  @staticmethod
  def load(path):
    '''
    从指定路径读取log
    '''
  
    with open(path, 'rb') as f:
      log = pickle.load(f)

    if type(log) is not Log:
      raise TypeError("不是日志文件")

    return log

