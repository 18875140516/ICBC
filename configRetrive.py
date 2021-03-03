import paho.mqtt.client as mqtt
import threading
import logging
import json
import time
# MQTT_URL = '211.67.21.65'
MQTT_URL = 'x.y.z.p'
PORT =1883
TOPIC = 'config'
logging.basicConfig(filename='logger.log', level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',)
class ConfigRetrive:
    #initialize config
    def __init__(self):
        self.config = dict()
        self.client = mqtt.Client()
        def task(client, config):
            def on_connect(client, userdata, flags, rc):
                client.subscribe(topic=TOPIC)
                logging.info('subscribe '+TOPIC +" OK!")
            def on_message(client, userdata, msg):
                logging.info(msg.payload)
                kv = json.loads(msg.payload)
                config[list(kv.keys())[0]] = list(kv.values())[0]
                logging.info(kv)
            client.on_connect = on_connect
            client.on_message = on_message
            client.connect(MQTT_URL, PORT, 60)
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

    
#构造配置生成器
bg = ConfigRetrive()
while True:
    #通过get函数获取对应的值，key需要获取值对应的键，default_value为该key对应的默认值（自己写一个同类型的）
    val = bg.get(key='name', default_value='xyz')
    #获取到的val有规范的格式，可根据此格式进行后续操作
    logging.info(val)
    time.sleep(5)

