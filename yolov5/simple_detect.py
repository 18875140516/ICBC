import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

global_model = None
class simple_yolov5:
    def __init__(self, weights='/home/liuyongzhi/data/yolov5x.pt', img_size=640):
        self.half = True
        self.imgsz = img_size
        self.device = select_device('')
        self.model = attempt_load(weights, map_location=self.device)
        if self.half:
            self.model.half()
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(img_size, s=stride)  # check img_size
        # self.model(torch.zeros(1, 3, img_size, img_size).to(self.device).type_as(next(self.model.parameters())))  # run once
        # global_model = attempt_load(weights, map_location=self.device)

        self.conf_thres, self.iou_thres = 0.25, 0.45
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # self.names = global_model.module.names if hasattr(global_model, 'module') else global_model.names
        self.frame = 0
        self.sum = 0
        self.fd = open('./log_simple', 'w')
        pass

    def detect_image(self, im0, classes=['person']):
        self.frame += 1
        # im0 = im0.astype("float16")
        img = self.__letterbox(im0, self.imgsz, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img = img/255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        print(img.shape)#512 640
        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)


        det = pred[0]
        if not len(det):
            return [], []
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        xywhs = []
        tlwhs = []
        classnames = []

        for *xyxy, conf, cls in reversed(det):
            xyxy = [int(i) for i in xyxy]
            if classes is not None and self.names[int(cls)] not in classes:
                continue
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            xywhs.append(xywh)
            tlwhs.append([xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]])
            classnames.append(self.names[int(cls)])
        return tlwhs, classnames

    def __letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)



if __name__ == '__main__':
    with torch.no_grad():
        model = simple_yolov5()
        cap = cv2.VideoCapture('/media/video/test.avi')
        while True:

            ret, img = cap.read()
            t1 = time.time()
            tlwhs, classes = model.detect_image(img)
            cp = img.copy()
            for tlwh, x in zip(tlwhs, classes):
                cv2.rectangle(cp, (tlwh[0], tlwh[1]), (tlwh[2]+tlwh[0], tlwh[3]+tlwh[1]), (255, 255,255))
            print((time.time() - t1) * 1000, 'ms')
            cv2.imshow('cp', cp)
            cv2.waitKey(20)

