import threading
import time
import pyaudio
import wave


class Recorder(threading.Thread):
    '''
    录音线程。
    一旦开启，一直录音。
    '''

    def __init__(
        self, outputPath, record_time=2.5,
        chunk=1024, fmt=pyaudio.paInt16,
        channels=1, sr=44100
    ):  
        '''
        自动录音分段
        ------
        + outputPath 保存文件目录
        + record_time 间隔时间
        '''
        super().__init__()

        self.outputPath = outputPath
        self.record_time = record_time
        self.chunk = chunk
        self.fmt = fmt
        self.channels = channels
        self.sr = sr

        if self.outputPath[-1] not in ('\\', '/'):
            self.outputPath += '\\'
        
        self.terminated = False
        
    def run(self):

        p = pyaudio.PyAudio()
        stream = p.open(format=self.fmt,
                        channels=self.channels,
                        rate=self.sr,
                        input=True,
                        frames_per_buffer=self.chunk)
        frames = []

        print('录音启动')
        while not self.terminated:
            fout = self.outputPath + "output_" +\
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())\
                + ".wav"
            frames = frames[-int(self.sr / self.chunk * self.record_time):]
            for i in range(0, int(self.sr / self.chunk * self.record_time)):
                data = stream.read(self.chunk)
                frames.append(data)
            wf = wave.open(fout, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(p.get_sample_size(self.fmt))
            wf.setframerate(self.sr)
            wf.writeframes(b''.join(frames))
            wf.close()

        stream.stop_stream()
        stream.close()
        p.terminate()
        print('录音结束')
    
    def terminate(self):
        
        self.terminated = True
