import paho.mqtt.publish as publish
import json
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
for key in root:
    cfg = dict()
    cfg[key] = root[key]
    s = json.dumps(cfg)
    publish.single(topic='config', payload=s, hostname="211.67.21.65")
