import cv2
import subprocess
import time
rtmp = r'rtmp://127.0.0.1:1935/hls/test2'

ret = 'http://127.0.0.1:8081/hls/test2.m3u8'
cap = cv2.VideoCapture('/media/video/test.avi')
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
sizeStr = str(size[0]) + 'x' + str(size[1])
command = ['ffmpeg',
           '-re',
            '-i', '-',
            '-f', 'flv',
            '-vprofile', 'baseline',
           '-vcodec', 'libx264',
           '-acodec', 'aac',
           '-ar', '44100',
           '-strict', '-2',
           '-ac', '-1',

            '-r', '5',
            '-s', '1280x720',
           '-q','10',
            # '-preset', 'ultrafast',
           rtmp]
pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)
idx = 1
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # time.sleep(0.2)
        # cv2.waitKey(40)
        for i in range(1000000):
            pass
        pipe.stdin.write(frame.tostring())
        print(idx)
        idx += 1
    else:
        cap = cv2.VideoCapture('/media/video/test.avi')
cap.release()
pipe.terminate()