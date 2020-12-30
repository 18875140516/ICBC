import sys
from yolo import YOLO
import cv2
import json
import os
from PIL import Image
from utils.util import checkPoint
import argparse
import subprocess
rtmp = r'rtmp://211.67.20.74:1935/myapp'
OK = True


def main(yolo, cfg):
    cap = cv2.VideoCapture(cfg['queueURL'])
    points = cfg['area']
    maxQueueSize = cfg['maxQueueSize']

    # def onMousecb(event, x, y, flags, params):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         print(checkPoint((x, y), points))
    # cv2.namedWindow('win')
    # cv2.setMouseCallback('win', onMousecb)
    idx = 0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sizeStr = str(size[0]) + 'x' + str(size[1])
    command = ['ffmpeg',
               '-y', '-an',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24',
               '-s', sizeStr,
               # '-r', '2',
               '-i', '-',
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               '-preset', 'ultrafast',
               '-f', 'flv',
               rtmp]
    pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)
    while OK:
        ret, frame = cap.read()

        if ret == False:
            cap = cv2.VideoCapture(cfg['queueURL'])
            ret, frame = cap.read()
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, classname = yolo.detect_image(image)  # xy w h
        cp = frame.copy()
        queueSize = 0
        for box in boxs:
            cpt = ((box[0]*2 + box[2])/2, box[1]+box[3])
            if checkPoint(cpt, points):
                cv2.rectangle(cp, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,0,255), 2)
                queueSize += 1
            else:
                cv2.rectangle(cp, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 255, 255), 2)
        for i in range(len(points)):
            cv2.line(cp, tuple(points[i]), tuple(points[(i+1)%len(points)]), (0, 0, 255), thickness=2)
        if queueSize >= maxQueueSize:
            cv2.putText(cp, 'too much perple', (200, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                        thickness=2)
        cv2.putText(cp,'queue: '+str(queueSize), (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), thickness=2)
        # cv2.imshow('win', cp)
        q = cv2.waitKey(30)
        if q == ord('q'):
            break
        elif q == ord('x'):
            cv2.imwrite('{:04d}.jpg'.format(idx), cp)

        pipe.stdin.write(cp.tostring())
        idx += 1
    pipe.terminate()
    pass


if __name__ == '__main__':
    with open('../config/queue.json', 'r') as f:
        cfg = json.load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=os.path.abspath('../model_data/yolo.h5'))
    parser.add_argument('--classpath', type=str, default=os.path.abspath('../model_data/coco_classes.txt'))
    parser.add_argument('--anchorpath', type=str, default=os.path.abspath('../model_data/yolo_anchors.txt'))
    args = parser.parse_args()
    main(YOLO(model_path=args.model,class_path=args.classpath, anchor_path=args.anchorpath), cfg)
