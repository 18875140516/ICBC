import time

import cv2
import subprocess
#用于hls推流，但是由于hls推流延迟过高，现在没有在使用
class RTMP_AGENT:
    def __init__(self, topic = 'test3',rtmp_url='211.67.21.65', rtmp_port='1935', protocal='hls'):
        self.rtmp_url = rtmp_url
        self.rtmp_port = rtmp_port
        self.topic = topic
        self.pipe = None
        self.protocal = protocal
        pass

    def send_image(self, frame):#
        assert frame is not None
        #write time h:m:s
        cv2.putText(frame, time.asctime(),(50,100), cv2.FONT_HERSHEY_PLAIN, fontScale=1,color=(255,255,255))
        if self.pipe is None:
            sizeStr = str(frame.shape[1])+'x' + str(frame.shape[0])
            m3u8_url = 'rtmp://' + self.rtmp_url + ':' + str(self.rtmp_port) + '/' + self.protocal + '/' + self.topic
            # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            # sizeStr = str(size[0]) + 'x' + str(size[1])
            print('sizeStr= ', sizeStr)
            command = ['ffmpeg',
                       '-y', '-an',
                       '-f', 'rawvideo',
                       '-vcodec', 'rawvideo',
                       '-pix_fmt', 'bgr24',
                       '-s', sizeStr,
                       '-r', '5',
                       '-re',
                       '-i', '-',
                       '-c:v', 'libx264',
                       '-pix_fmt', 'yuv420p',
                       '-preset', 'ultrafast',
                       '-f', 'flv',
                       m3u8_url]
            self.pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)
            assert frame is not None
        self.pipe.stdin.write(frame.tostring())

    def __del__(self):
        self.pipe.terminate()
#---------test mode----------
# import sys
# video_path = '/media/video/test.avi'
# topic = 'test'
# if len(sys.argv) == 2:
#     topic = sys.argv[1]
# elif len(sys.argv) == 3:
#     topic = sys.argv[1]
#     video_path = sys.argv[2]
# print('topic = ', topic)
# print('video = ', video_path)
# cap = cv2.VideoCapture(video_path)
# ret, img = cap.read()
# if ret == False:
#     exit()
#
# agent = RTMP_AGENT(topic=topic)
# while ret:
#     agent.send_image(img)
#     ret, img = cap.read()
#     if ret == False:
#         cap = cv2.VideoCapture(video_path)
#         ret, img = cap.read()
