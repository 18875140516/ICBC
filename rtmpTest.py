import sys
import cv2
from rtmpAgent import RTMP_AGENT
video_path = '/media/video/ch35_.mp4'
topic = 'test'
if len(sys.argv) == 2:
    topic = sys.argv[1]
elif len(sys.argv) == 3:
    topic = sys.argv[1]
    video_path = sys.argv[2]
print('topic = ', topic)
print('video = ', video_path)
cap = cv2.VideoCapture(video_path)
ret, img = cap.read()
if ret == False:
    exit()

agent = RTMP_AGENT(topic=topic, protocal='rtmplive')
while ret:
    agent.send_image(img)
    ret, img = cap.read()
    if ret == False:
        cap = cv2.VideoCapture(video_path)
        ret, img = cap.read()
