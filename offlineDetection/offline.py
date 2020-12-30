#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import sys
print(sys.path)
sys.path.append('../')
import argparse
import json
import os
import time
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from mgn.network import MGN
# from coordinate_transform import CoordTrans
from mgn.utils.extract_feature import extract_feature
from yolo import YOLO
from utils.util import checkPoint
import paho.mqtt.publish as publish
# from debug_cost_mat import *
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

startTrack = False
tracked = False
clickPoint = None
def load_model_pytorch(model_path):
    model = MGN()
    model = model.to('cuda')
    model.load_state_dict(torch.load(model_path))
    return model

# def get_feature(model, img_path):
#     model.eval()
#     test_transform = transforms.Compose([
#         transforms.Resize((384, 128), interpolation=3),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     features = torch.FloatTensor()
#     for i in img_path:
#         query_image = test_transform(default_loader(i))
#         query_feature = extract_feature(model, tqdm([(torch.unsqueeze(query_image, 0), 1)]))
#         features = torch.cat((features, query_feature), 0)
#     return features

#保存检测过程中的bbox
def box_encode(model, img, boxes, prefix='./img1/'):

    imgs = []
    for box in boxes:
        _box = box.copy()
        _box[2] = _box[0] + _box[2]
        _box[3] = _box[1] + _box[3]
        imgs.append(img.crop(_box))

    # img.save(os.path.join(prefix, 'raw.jpg'), quality=95)
    # for idx, x in enumerate(imgs):
    #     x.save(os.path.join(prefix, 'box'+str(idx)+'.jpg'), quality=95)

    model.eval()
    features = torch.FloatTensor()
    test_transform = transforms.Compose([
        transforms.Resize((384, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for _img in imgs:
        _img = test_transform(_img)
        query_feature = extract_feature(model, [(torch.unsqueeze(_img, 0), 1)])
        features = torch.cat((features, query_feature), 0)
    return features.numpy()


def main(yolo, args, cfg):  # 输入yolov3模型和视频路径
    global startTrack
    global tracked
    global clickPoint
    # Definition of the parameters
    max_cosine_distance = 0.3  # 允许相同人的最大余弦距离0.3
    nn_budget = None
    nms_max_overlap = 1.0  # 非极大值抑制，减少重复bbox，针对一类物体独立操作
    # deep_sort
    model_filename = 'model_data/mars-small128.pb'  # 128维特征预测模型，效果不佳，rank1极低
    # encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # mgn model
    print('loading mgn model')
    mgn = load_model_pytorch('/home/lyz/Desktop/ReID-MGN/model.pt')
    print('load mgn OK')
    test_video_flag = True
    writeVideo_flag = False  # 是否写入视频


    cv2.namedWindow('win')


    def clickCB(event, x, y, flag, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            global startTrack
            global clickPoint
            global tracked
            startTrack = True
            clickPoint = (x, y)
            tracked = False
    cv2.setMouseCallback('win',clickCB)


    fps = 0.0

    cap = cv2.VideoCapture(args.input)

    idx = 1
    metric = nn_matching.NearestNeighborDistanceMetric("mgn", 0.3, nn_budget)
    target = None
    while True:
        # print(clickPoint)
        ret, img = cap.read()
        if ret == False:
            print('read error')
            break
        boxs, classname = yolo.detect_image(Image.fromarray(img[...,::-1]))
        features = box_encode(mgn, Image.fromarray(img[..., ::-1]), boxs, prefix='./img1')
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        if tracked == False and startTrack == True: #choose detection
            # print('1')
            for det in detections:
                bbox = det.tlwh.copy()
                bbox[2:] += bbox[:2]
                if clickPoint[0] > bbox[0] \
                    and clickPoint[0] < bbox[2]\
                    and clickPoint[1] > bbox[1]\
                    and clickPoint[1] < bbox[3]:
                    print('found it')
                    target = det
                    tracked = True
                    break
        elif startTrack  and tracked :#filter extra detection
            dis = []
            for i, det in enumerate(detections):
                bbox = det.tlwh.copy()
                bbox[2:] += bbox[:2]
                dis.append((i, np.dot(target.feature, det.feature)))

            dis.sort(key=lambda x: x[1], reverse=True)
            if dis[0][1] <  0.85:
                continue
            # print(len(dis), dis)
            bbox = detections[dis[0][0]].tlwh.copy()
            bbox[2:] += bbox[:2]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)

            pass
        else:
            for id, det in enumerate(detections):
                bbox = det.tlwh.copy()
                bbox[2:] += bbox[:2]
                # print(bbox)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)

        cv2.imshow('win', img)
        cv2.waitKey(20)
        # pass
    cv2.destroyAllWindows()

if __name__ == '__main__':
    with open('../config/standing.json', 'r') as r:
        cfg = json.load(r)
    print(cfg)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=os.path.abspath('/media/video/test.avi'))
    parser.add_argument('--cfg-pa', type=str, default=os.path.abspath('../config'))
    parser.add_argument('--use-model', type=bool, default=True)


    args = parser.parse_args()
    main(YOLO(), args, cfg)


