# 方便使用的连接建立
# 参考自： https://blog.csdn.net/u014022631/article/details/88670780

import socket
import sys
import base64
import hashlib
import struct

# CONFIG #
LOCALHOST = b'localhost'
MAGIC_STRING = b'258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
HANDSHAKE_STRING =\
      b"HTTP/1.1 101 Switching Protocols\r\n" \
      b"Upgrade:websocket\r\n" \
      b"Connection: Upgrade\r\n" \
      b"Sec-WebSocket-Accept: {1}\r\n" \
      b"WebSocket-Location: ws://{2}/chat\r\n" \
      b"WebSocket-Protocol:chat\r\n\r\n"


class Handler:
    '''
    处理一个连接活动的类

    --------
    Cite from:

    '''

    def __init__(self, connection):

        self.con = connection
        
    def recv_data(self, num):

        all_data = self.con.recv(num)
        if not len(all_data):
            return False
        else:
            print(all_data[1])
            code_len = all_data[1] & 127

        if code_len == 126:
            masks = all_data[4:8]
            data = all_data[8:]
        elif code_len == 127:
            masks = all_data[10:14]
            data = all_data[14:]
        else:
            masks = all_data[2:6]
            data = all_data[6:]

        raw_str = ""
        i = 0
        for d in data:
            raw_str += chr(d ^ masks[i % 4])
            i += 1
   
        return raw_str
 
    def send_data(self, data):

        if data:
            pass
        else:
            return False

        token = b"\x81"
        length = len(data)
        if length < 126:
            token += struct.pack("B", length)
        elif length <= 0xFFFF:
            token += struct.pack("!BH", 126, length)
        else:
            token += struct.pack("!BQ", 127, length)
        # struct为Python中处理二进制数的模块，二进制流为C，或网络流的形式。

        data = b'%s%s' % (token, data)
        self.con.send(data)
        return True


def handshake(con, host, port):

    headers = {}
    shake = con.recv(1024)
    if not len(shake):
        return False
    header, data = shake.split(b'\r\n\r\n', 1)
    for line in header.split(b'\r\n')[1:]:
        key, val = line.split(b': ', 1)
        headers[key] = val
    if b'Sec-WebSocket-Key' not in headers:
        print('This socket is not websocket, client close.')
        con.close()
        return False
 
    sec_key = headers[b'Sec-WebSocket-Key']
    res_key = base64.b64encode(hashlib.sha1(sec_key + MAGIC_STRING).digest())
    str_handshake = HANDSHAKE_STRING.replace(b'{1}', res_key).replace(
        b'{2}', host + b':' + str.encode(str(port)))
    
    print(str.encode(str(port)))
    con.send(str_handshake)
    return True


def connectAcpt(port, host=LOCALHOST, backlog=1000):
    """
    在等待连接到来。
    """
 
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.listen(backlog)
        print(f"bind {port}, ready to use")
    except:
        print("无法创建新服务")
        sys.exit()
 
    while True:
        connection, address = sock.accept()  # wait for connection
        print(f"从{address}取得连接：", end='')
        if handshake(connection, host, port):
            print("握手成功")
            try:
                handler = Handler(connection)
                print('连接已创建')
                return handler
            except:
                print('连接创建失败')
                connection.close()
 
 