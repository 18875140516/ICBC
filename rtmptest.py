import cv2
import subprocess
import time
rtmp = r'rtmp://127.0.0.1:1935/myapp'
rtmp2 = r'rtmp://127.0.0.1:1935/hls/test2'
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
           rtmp2]
pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)
idx = 1
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # time.sleep(0.5)
        # cv2.waitKey(40)
        cv2.putText(frame, str(idx), (50,50), cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0,0,255), thickness=3)
        # cv2.imshow('img', frame)
        # cv2.waitKey(20)
        pipe.stdin.write(frame.tostring())
        print(idx)
        idx += 1
    else:
        cap = cv2.VideoCapture('/media/video/test.avi')
cap.release()
pipe.terminate()