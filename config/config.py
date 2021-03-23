import paho.mqtt.publish as publish
import json
import sys
sys.path.append('../')
root = dict()
root = {
  "MQTT_URL":"211.67.21.65",
  "MQTT_PORT":  1883,
  "MYSQL_URL": "211.67.21.65",
  "MYSQL_PORT": 3306,
  "RTMP_URL": "211.67.21.65",
  "RTMP_PORT": 1935,
  "CONFIG_SERVER_URL": "211.67.21.65",
  "CONFIG_SERVER_PORT": 10086,
  "STANDING_TOPIC": "standing",
  "DEAULT_AREA": [[0.3770833333333333, 0.4515625],
 [0.46458333333333335, 0.6671875],
 [1.29375, 0.628125],
 [1.0416666666666667, 0.4140625]],
  "VIDEO_PATH": "/media/video/test.avi",
  "OUTPUT_PATH":"/media/video/output.mp4",
  "WARNING_CYCLE":"5s",
  "SHOW_BBOX": True,
  "SHOW_PATH": False,
  "SHOW_AREA": True

}
# for key in root:
#     cfg = dict()
#     cfg[key] = root[key]
#     s = json.dumps(cfg)
#     publish.single(topic='config', payload=s, hostname="211.67.21.65")


#-------------------test publish config---------------
#import cv2
#import base64
#cap = cv2.VideoCapture('/media/video/test.avi')
#MQTT_URL = '211.67.21.65'
##send to config
#ret, img = cap.read()
#assert ret == True
#img = cv2.resize(img, (348, 128))
#img_b64 = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
#
#print()
#print()
#topic = 'standing_area'
#publish.single(topic='config',payload=json.dumps({topic:img_b64}),hostname=MQTT_URL)
#
##---------------------test get config --------------------------
#import time
#time.sleep(3)
#from configRetrive import ConfigRetrive
#config = ConfigRetrive()
