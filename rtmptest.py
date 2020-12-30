import cv2
import subprocess
import time
rtmp = r'rtmp://211.67.20.74:1935/myapp'
cap = cv2.VideoCapture('/media/video/test.avi')
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
sizeStr = str(size[0]) + 'x' + str(size[1])
command = ['ffmpeg',
           '-y', '-an',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', sizeStr,
           '-r', '5',
           '-i', '-',
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
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