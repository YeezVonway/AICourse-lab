import torch
import numpy as np
import librosa
import utils
import os
import threading

import CONFIG
import models


def get_evalModel(cfg=CONFIG.GLOBAL):
    
    return models.get_model(cfg.FILE.EVAL_MODEL, cfg)


def eval_file(
    model,
    audioFile,
    cfg=CONFIG.GLOBAL
):

    try:
        arr, _ = librosa.load(audioFile, sr=cfg.DATA.SR)
    except:
        return None

    arr = utils.get_frames(arr, cfg.DATA)
    arr = np.expand_dims(arr, 0)
    x = torch.tensor(
        arr, dtype=torch.float32,
        device=torch.device(cfg.DEVICE)
        )
    y: torch.Tensor = model(x)

    return y.squeeze(dim=0).cpu().detach().numpy()


def save_result(y, dst):

    tempDst = dst + '.temp'
    with open(tempDst, 'w') as dstFile:
        dstFile.write(str(int(np.argmax(y))))
        dstFile.write('\n')
        for val in y:
            dstFile.write(str(float(val)) + ' ')
        dstFile.write('\n')

    if(os.path.exists(dst)):
        os.remove(dst)
    os.rename(tempDst, dst)
    

if __name__ == "__main__":
    '''
    开始预测界面。
    '''

    import tkinter as tk
    from tkinter import messagebox
    import shutil
    import time
    
    torch.set_grad_enabled(False)
    root = tk.Tk()
    root.resizable(False, False)

    row = 0
    tk.Label(root, text="输入音频目录").grid(row=row, column=0, padx=5, pady=5)
    entry_inputDir = tk.Entry(root, width=32)
    entry_inputDir.insert(tk.END, "./eval-input")
    entry_inputDir.grid(row=row, column=1, columnspan=3, padx=5, pady=5)

    row += 1
    tk.Label(root, text="输出结果目录").grid(row=row, column=0, padx=5, pady=5)
    entry_outputDir = tk.Entry(root, width=32)
    entry_outputDir.insert(tk.END, "./eval-output")
    entry_outputDir.grid(row=row, column=1, columnspan=3, padx=5, pady=5)

    row += 1
    btn_run = tk.Button(root, text="运行服务", width=12)
    btn_terminate = tk.Button(root, text="中止服务", width=12, state=tk.DISABLED)

    running = False
    thread = None

    def thread_run_server(inputDir, outputDir, evaledDir):
        global running

        print("服务启动中")
        running = True
        model = get_evalModel(CONFIG.GLOBAL)
        model.eval()
        print("服务正在运行")

        while(running):
            files = os.listdir(inputDir)
            for fname in files:
                if fname == 'evaled':
                    continue
                inputFile = inputDir + fname
                y = None
                while y is None:
                    y = eval_file(model, inputDir + fname, CONFIG.GLOBAL)
                    if y is None:
                        time.sleep(0.1)
                        print(f"读取{fname}失败，将尝试重新读取。")
                    if not running: 
                        break
                if not running: 
                    break
                outputFile = os.path.splitext(fname)[0] + '.evl'
                save_result(y, outputDir + outputFile)
                if(os.path.exists(evaledDir + fname)):
                    os.remove(evaledDir + fname)
                shutil.move(inputFile, evaledDir)
                print(f"已预测：{fname},输出至：{outputFile}")
            if len(files) <= 1:
                time.sleep(0.1)
        
        del model

    def cmd_run_server():

        global running, btn_run, thread

        inputDir = entry_inputDir.get()
        outputDir = entry_outputDir.get()
        if inputDir[-1] not in ('\\', '/'):
            inputDir += '\\'
        if outputDir[-1] not in ('\\', '/'):
            outputDir += '\\'
        evaledDir = inputDir + 'evaled\\'

        if not os.path.exists(inputDir):
            messagebox.showerror('错误', '输入目录不存在！')
            return
        if not os.path.exists(outputDir):
            yes = messagebox.askyesno(
                '创建新目录',
                '输出目录不存在，是否创建?')
            if yes:
                try:
                    os.mkdir(outputDir)
                except:
                    messagebox.showerror('错误', '无法创建指定目录')
                    return
            else:
                return
        if not os.path.exists(evaledDir):
            os.mkdir(evaledDir)

        thread = threading.Thread(
            target=thread_run_server,
            args=(inputDir, outputDir, evaledDir),
            daemon=True)

        btn_run.config(state=tk.DISABLED)
        btn_terminate.config(state=tk.NORMAL)
        root.update()

        thread.start()

    btn_run.config(command=cmd_run_server)
    btn_run.grid(row=row, column=0, columnspan=2)

    def cmd_term_server():
        global running, btn_run, btn_terminate
        running = False
        thread.join()
        btn_run.config(state=tk.NORMAL)
        btn_terminate.config(state=tk.DISABLED)
        root.update()
        print("服务已终止")

    btn_terminate.config(command=cmd_term_server)
    btn_terminate.grid(row=row, column=2, columnspan=2)

    tk.mainloop()

