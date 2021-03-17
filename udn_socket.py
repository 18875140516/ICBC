import socket
import numpy as np
import time
import base64
import os
import threading
import cv2
import logging
# logging.basicConfig(filename='logger.log', level=logging.INFO,
#                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                     datefmt='%a, %d %b %Y %H:%M:%S')
logging.basicConfig(filename='logger.log', level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
SERVER_ADDR = 'test'
logging.info('-'*50)
def client():
    time.sleep(1)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if sock.fileno() < 0:
        logging.info('socket error')
    try:
        sock.connect(SERVER_ADDR)
    except socket.error:
        logging.info('error')
    PATH = '/media/video/test.avi'
    cap = cv2.VideoCapture(PATH)
    while True:
        ret, img = cap.read()
        if ret == False:
            cap = cv2.VideoCapture(PATH)
            ret, img = cap.read()
        # print(ret, img.shape)
        img_b64 = base64.b64encode(cv2.imencode('.jpg', img)[1])

        len_code = '{:07d}'.format(len(img_b64))
        print('send sz = ',  str(len(len_code)))
        logging.info('send len ' + str(len(len_code)))
        sock.sendall(len_code.encode('utf8'))
        logging.info('send data :' + str(len(img_b64)))
        sock.sendall(img_b64)
        # time.sleep(0.1)

    pass

def server():
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if sock.fileno() < 0:
        logging.info('socket error')
    if os.path.exists(SERVER_ADDR):
        os.unlink(SERVER_ADDR)

    if sock.bind(SERVER_ADDR):
        logging.info('bind error')

    if sock.listen(5):
        logging.info('listen error')

    while True:
        conn, client_addr = sock.accept()
        logging.info('a new socket ')
        try:
            idx = 0
            while True:
                logging.info('receive len:7')
                sz = conn.recv(7)
                if sz:
                    print('rec sz = ', sz)
                    logging.info('receive data :' + str(sz))
                    data = conn.recv(int(sz))
                    img_b64 = base64.b64decode(data)
                    logging.info('server' + str(len(img_b64)))
                    # print(img_b64)
                    img_array = np.fromstring(img_b64, np.uint8)
                    img = cv2.imdecode(img_array,cv2.COLOR_BGR2RGB)
                    if not os.path.exists('./tmp'):
                        os.mkdir('./tmp')
                    cv2.imwrite('./tmp/{:04d}.jpg'.format(idx), img)
                    idx+=1
                    # print(img)
                    # cv2.imshow('img', img)
                    # cv2.waitKey(30)

        finally:
            conn.close()
    os.unlink(SERVER_ADDR)

    pass

class UDNClient:
    def __init__(self, server_addr='/tmp/test_file'):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        print(self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF))
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1111111)
        # self.sock.setsockopt(socket.SOL_SOCKET, socket.TCP_NODELAY, True)
        assert self.sock.fileno() >= 0
        try:
            self.sock.connect(SERVER_ADDR)
        except socket.error:
            print('error')

        pass
    def send_img(self, img):
        img_b64 = base64.b64encode(cv2.imencode('.jpg', img)[1])
        len_code = '{:07d}'.format(len(img_b64))
        print('send sz = ',  str(len(len_code)), ' ', len_code)
        self.sock.sendall(len_code.encode('utf8'))
        self.sock.sendall(img_b64)

class UDNServer:
    def __init__(self, servr_addr='/tmp/test_file'):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,  1111111)
        print(self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
        assert self.sock.fileno() >= 0
        if os.path.exists(SERVER_ADDR):
            os.unlink(SERVER_ADDR)
        if self.sock.bind(SERVER_ADDR):
            logging.info('bind error')

        if self.sock.listen(5):
            logging.info('listen error')

    def transfer_img(self, wsclient):
        while True:
            conn, client_addr = self.sock.accept()
            logging.info('a new socket ')
            try:
                idx = 0
                while True:
                    logging.info('receive len:7')
                    sz = conn.recv(7)
                    if sz:
                        print('rec sz = ', sz)
                        sz = int(sz)
                        logging.info('receive data :' + str(sz))
                        data = conn.recv(int(sz))
                        while len(data) < sz:

                            sz -= len(data)
                            print('-' * 50, ' res=', sz)
                            data += conn.recv(int(sz))

                        img_b64 = base64.b64decode(data)
                        if wsclient is not None:
                            wsclient.send(img_b64)
                        logging.info('server' + str(len(img_b64)))
                        # print(img_b64)
                        img_array = np.fromstring(img_b64, np.uint8)
                        img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
                        if not os.path.exists('./tmp'):
                            os.mkdir('./tmp')
                        cv2.imwrite('./tmp/{:04d}.jpg'.format(idx), img)
                        idx += 1
                        # print(img)
                        # cv2.imshow('img', img)
                        # cv2.waitKey(30)

            finally:
                conn.close()
        os.unlink(SERVER_ADDR)

        pass

# def main():
#     t_server = threading.Thread(target=server)
#
#     t_server.start()
#
#     t_client = threading.Thread(target=client)
#     t_client.start()
#
#     t_server.join()
#     t_client.join()
#
#
#     pass
#
# def main2():
#     client = UDNClient()
#
#
#     pass
# if __name__ == '__main__':
#     main2()