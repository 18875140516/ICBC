from yolo import YOLO
import cv2
import json
import os
from PIL import Image
from util.util import checkPoint

OK = True


def main(yolo, cfg):
    cap = cv2.VideoCapture(cfg['queueURL'])
    points = cfg['area']
    maxQueueSize = cfg['maxQueueSize']

    def onMousecb(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(checkPoint((x, y), points))
    cv2.namedWindow('win')
    cv2.setMouseCallback('win', onMousecb)

    while OK:
        ret, frame = cap.read()
        if ret == False:
            break
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
            cv2.putText(cp, 'too much perple', (200, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255),
                        thickness=2)
        cv2.putText(cp,'queue: '+str(queueSize), (50,50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,255,255), thickness=2)
        cv2.imshow('win', cp)
        q = cv2.waitKey(30)
        if q == ord('q'):
            break
    pass


if __name__ == '__main__':
    with open('../config/queue.json', 'r') as f:
        cfg = json.load(f)
    print(cfg)
    main(YOLO(), cfg)
    pass