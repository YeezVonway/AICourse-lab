# 部署服务

import numpy as np
import os
import tkinter as tk
from tkinter import messagebox
import shutil
import time
import threading
import torch
import json
import audio_classification as ac
from record import Recorder
import easyconn


def toJson(prediction, time_create):
    '''
    获得json格式输出 \n
    ------
    + prediction: 各个类型得分
    + time_create: 音频文件的创建时间缀
    '''

    result = {'scores': {}}
    for i in range(ac.CONFIG.GLOBAL.DATA.NUM_CLASS):
        clsName, _ = ac.CONFIG.GLOBAL.DATA.CLASSES[i]
        score = float(prediction[i])
        result['scores'][clsName] = score
    
    result['class'], _ =\
        ac.CONFIG.GLOBAL.DATA.CLASSES[np.argmax(prediction)]
    result['time'] = time_create
    
    return json.dumps(result).encode('UTF-8')


class Service(threading.Thread):

    def __init__(self, port, cacheDir, deleteEvaled=True):
        '''
        录音和分类服务线程\n
        ------
        + port: 本地端口号
        + cacheDir: 录音的缓存文件
        + deleteEvaled: 当True时，删除已经预测类别的录音。
          否则，已经分类的音频将保留到cacheDir/evaled目录中
        '''
        
        super().__init__()
        self.terminated = False
        self.running = False
        self.port = port
        self.cacheDir = cacheDir
        self.evaledDir = cacheDir + 'evaled\\'
        self.deleteEvaled = deleteEvaled
        
    def run(self):

        # 清除缓存
        files = os.listdir(self.cacheDir)
        for fname in files:
            if fname == "evaled":
                continue
            cachedFile = self.cacheDir + fname
            if self.deleteEvaled:
                os.remove(cachedFile)
            else:
                if(os.path.exists(self.evaledDir + fname)):
                    os.remove(self.evaledDir + fname)
                shutil.move(cachedFile, self.evaledDir)

        print("服务已部署，等待连接中")
        handler = easyconn.connectAcpt(self.port)
        recoder = Recorder(self.cacheDir)
        recoder.start()
        self.running = True
        print("连接已建立，正在提供分类服务")

        while(self.terminated is False):

            files = os.listdir(self.cacheDir)
            for fname in files:  # 清空整个文件夹
                
                # 获得一个预测
                if fname == 'evaled':  # 忽略evaled子目录
                    continue
                inputFile = self.cacheDir + fname
                y = None
                try:  # 预测
                    y = ac.evaluate.eval_file(model, inputFile)
                except ValueError:
                    print(f"音频{fname}读取失败")
                except FileNotFoundError:
                    print(f"音频{fname}读取失败")
                    

                if y is not None:
                    jstr = toJson(y, os.path.getctime(inputFile))
                    try:
                        handler.send_data(jstr)
                    except ConnectionAbortedError:
                        self.terminated = True
                        break
                    print(f"自{fname}预测: {jstr}")
                if self.deleteEvaled and os.path.exists(inputFile):
                    os.remove(inputFile)
                else:
                    if(os.path.exists(self.evaledDir + fname)):
                        os.remove(self.evaledDir + fname)
                    shutil.move(inputFile, self.evaledDir)
            
            if len(files) <= 1:  # 目录空，等待
                time.sleep(0.1)

        #  结束
        recoder.terminated = True
        recoder.join()
        self.running = False
        handler.con.close()
        del handler
        print("服务终止")


# CONFIG #
DEFAULT_ADDR = 'localhost'
DEFAULT_PORT = '3368'
DEFAULT_INPUTDIR = '.\\audio_input'

if __name__ == "__main__":

    global_service = None
    torch.set_grad_enabled(False)
    model = ac.evaluate.get_evalModel()
    model.eval()

    root = tk.Tk()
    root.resizable(False, False)

    row = 0
    tk.Label(root, text="输入音频目录").grid(row=row, column=0, padx=5, pady=5)
    entry_cacheDir = tk.Entry(root, width=32)
    entry_cacheDir.insert(tk.END, DEFAULT_INPUTDIR)
    entry_cacheDir.grid(row=row, column=1, columnspan=3, padx=5, pady=5)

    row = 1
    tk.Label(root, text="服务端口").grid(row=row, column=0, padx=5, pady=5)
    entry_port = tk.Entry(root, width=32)
    entry_port.insert(tk.END, DEFAULT_PORT)
    entry_port.grid(row=row, column=1, columnspan=3, padx=5, pady=5)

    row = 2
    opt_delEvl = tk.BooleanVar(value=True)
    tk.Checkbutton(
        root, text='删除已分类的录音缓存', 
        onvalue=True, offvalue=False, variable=opt_delEvl,
        ).grid(row=row, column=0, columnspan=2, padx=5, pady=5)
    btn_run = tk.Button(root, text="启动", width=10)
    btn_run.grid(row=row, column=2, padx=5, pady=5)
    btn_terminate = tk.Button(root, text="终止", width=10, state=tk.DISABLED)
    btn_terminate.grid(row=row, column=3, padx=5, pady=5)

    def cmd_run_server():
        global global_service

        if global_service is not None:
            raise RuntimeError

        cacheDir = entry_cacheDir.get()
        port = int(entry_port.get())
        delEvl = opt_delEvl.get()
        if cacheDir[-1] not in ('\\', '/'):
            cacheDir += '\\'
        evaledDir = cacheDir + 'evaled\\'

        if os.path.exists(cacheDir) is False:
            choice = messagebox.askyesno('提示', '缓存目录不存在，是否创建')
            if choice is True:
                try:
                    os.mkdir(cacheDir)
                except FileNotFoundError:
                    messagebox.showerror('提示', '无法创建缓存目录')
                    return
            else:
                return
    
        if delEvl is not True and not os.path.exists(evaledDir):
            os.mkdir(evaledDir)

        global_service = Service(port, cacheDir, delEvl)
        global_service.start()
        while global_service.running is False:
            time.sleep(0.1)

        btn_run.config(state=tk.DISABLED)
        btn_terminate.config(state=tk.NORMAL)
        root.update()

    def cmd_term_server():
        global global_service

        if global_service is None:
            raise RuntimeError

        global_service.terminated = True
        global_service.join()
        global_service = None

        btn_run.config(state=tk.NORMAL)
        btn_terminate.config(state=tk.DISABLED)
        root.update()       

    btn_run.config(command=cmd_run_server)
    btn_terminate.config(command=cmd_term_server)

    tk.mainloop()

