import paho.mqtt.client as mqtt
import threading
import logging
import json
import time
import socket
MQTT_URL = '211.67.21.65'
# MQTT_URL = 'x.y.z.p'
PORT =1883
TOPIC = 'config'
#SERVER_IP='211.67.21.65'
SERVER_IP = '127.0.0.1'
SERVER_PORT = 10086
logging.basicConfig(filename='/log/logger.log', level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
class ConfigRetrive:
    #initialize config
    def __init__(self, url=MQTT_URL, port=PORT, topic=TOPIC):
        self.topic = topic
        self.port = port
        self.url = url
        self.config = dict()
        self.client = mqtt.Client()
        self.getConfigFromServer()
        def task(client, config):
            def on_connect(client, userdata, flags, rc):
                client.subscribe(topic=self.topic)
                logging.info('subscribe '+self.topic +" OK!")
            def on_message(client, userdata, msg):
                logging.info('get new config: ' + str(msg.payload))
                kv = json.loads(msg.payload)
                config[list(kv.keys())[0]] = list(kv.values())[0]
            client.on_connect = on_connect
            client.on_message = on_message
            client.connect(self.url, self.port, 60)
            client.loop_forever()

        self.thread_ = threading.Thread(target=task, args=(self.client, self.config))
        self.thread_.start()

    #get the config by key
    def get(self, key, default_value):
        if key in self.config.keys():
            return self.config[key]
        else:
            logging.warning(key + " not in config map" )
            return default_value

    def getConfigFromServer(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('connecting')
        s.connect((SERVER_IP, SERVER_PORT))
        print('connected')
        cfg_byte = s.recv(1024*1024)
        print(cfg_byte)
        print('parsing')
        cfg_str = cfg_byte.decode()
        self.config = json.loads(cfg_str)
        print(self.config)
#-------------------------------------------以下为操作示例--------------------    
# config = ConfigRetrive()
# while True:
#     print(config.config)
#     time.sleep(3)
# #模拟配置生,这个主要测试配置的生成,后续会实现自动化配置上传
# import paho.mqtt.publish as publish
# import numpy as np
# import threading
# area_cfg = dict()
# def task_thread():
#     while True:
#         #print('-')
#         test = int(np.random.randint(0,10, 1)[0])
#         area_cfg['face_area'] = [[0,0] , [1,1], [test, test]]
#         #print('face_area = ', area_cfg)
#         s = json.dumps(area_cfg)
#         # print(s)
#         publish.single(topic='config', hostname=MQTT_URL, payload=s)
#         time.sleep(5)
# t = threading.Thread(target=task_thread)
# t.start()
# #构造配置生成器
# bg = ConfigRetrive()
# while True:
#     #通过get函数获取对应的值，key需要获取值对应的键，default_value为该key对应的默认值（自己写一个同类型的）
#     val = bg.get(key='standing_area', default_value='xyz')
#     val2 = bg.get(key='test_cfg', default_value = 'test_cfg')
#     #获取到的val有规范的格式，可根据此格式进行后续操作
#     logging.info(val)
#     # logging.info(val2)
#     print("staning_area = ", val)
#     # print('val2 = ', val2)
#     time.sleep(3)
# t.join()
