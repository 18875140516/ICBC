#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import sys
# print(sys.path)
sys.path.append('../')
import argparse
import json
import os
import time
import warnings
import sys
sys.path.append(os.path.abspath(os.getcwd()))
# print(sys.path)
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
from utils.util import checkPoint, compute_iou
import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt
import threading
import json
# from debug_cost_mat import *

DEBUG_MODE = True

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
MQTT_URL = '127.0.0.1'
startTrack = False
tracked = False
clickPoint = None
WIDTH = None
HEIGHT = None
LAST_APPEAR = None  # 记录上一次出现的时间
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
    global HEIGHT
    global WIDTH
    global LAST_APPEAR
    managerStat = 'online'
    if not DEBUG_MODE:
        def listenMQ():
        #从消息队列获得信息，然后改变当前状态
            def on_connect(client, userdata, flags, rc):
                print("Connected with result code " + str(rc))

                # Subscribing in on_connect() means that if we lose the connection and
                # reconnect then subscriptions will be renewed.
                client.subscribe(topic='selectPerson')
                print('subscribe selectPerson successfully')

            # The callback for when a PUBLISH message is received from the server.
            def on_message(client, userdata, msg):
                global startTrack
                global tracked
                global clickPoint
                global WIDTH
                global HEIGHT
                global LAST_APPEAR
                startTrack = True
                print('reveive msg', msg.payload)
                ret = json.loads(msg.payload)
                x = int(WIDTH*float(ret['x']))
                y = int(HEIGHT*float(ret['y']))
                print('x=',x, 'y=',y)
                clickPoint = (x, y)
                LAST_APPEAR = time.time()
                tracked = False
                print(msg.payload)

            client = mqtt.Client()
            client.on_connect = on_connect
            client.on_message = on_message
            print('starting connecting')
            ret = client.connect(MQTT_URL, 1883, 60)
            print(ret)
            print('ending connecting')
            client.loop_forever()
        sub_thread = threading.Thread(target=listenMQ)
        sub_thread.start()

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

    if DEBUG_MODE:
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
    HEIGHT = cap.get(4)
    WIDTH = cap.get(3)
    idx = 1
    target = None
    records = []
    old_bbox = None
    old_lost_time = 0
    while True:
        ret, img = cap.read()
        img = cv2.resize(img, dsize=(720, 480))
        # img = cv2.resize(img, ( img.shape[1]//2,img.shape[0]//2))
        if ret == False:
            print('read OK')
            cap = cv2.VideoCapture(args.input)
            ret, img = cap.read()
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
                records.append(det.feature)
                if clickPoint[0] > bbox[0] \
                    and clickPoint[0] < bbox[2]\
                    and clickPoint[1] > bbox[1]\
                    and clickPoint[1] < bbox[3]:
                    print('found it')
                    old_bbox = bbox
                    target = det
                    tracked = True
                    break

        elif startTrack and tracked: #filter extra detection
            dis = []
            for i, det in enumerate(detections):
                bbox = det.tlwh.copy()
                bbox[2:] += bbox[:2]
                dis.append((i, np.dot(target.feature, det.feature)))

            dis.sort(key=lambda x: x[1], reverse=True)
            matched = False
            for i in range(len(dis)):
                #check if near old_bbox
                check_id = dis[i][0]
                cmp_bbox = detections[check_id].tlwh.copy()
                cmp_bbox[2:] += cmp_bbox[:2]

                def check_if_append(detections, check_id):
                    # print('checking')
                    MARGIN = 0
                    check_box = detections[check_id].tlwh.copy()
                    check_box[2:] += check_box[:2]
                    check_box = [check_box[0] - MARGIN, check_box[1] - MARGIN, check_box[2] + MARGIN,
                                 check_box[3] + MARGIN]
                    for i, det in enumerate(detections):
                        if i == check_id:
                            continue
                        _bbox = det.tlwh.copy()
                        _bbox[2:] += _bbox[:2]
                        iou = compute_iou(_bbox, check_box)
                        # print(iou)

                    pass

                if old_bbox is not None and compute_iou(old_bbox, cmp_bbox) < 0.01:
                    continue

                if dis[i][1] >= 0.80:

                    #consider appending records when closed det appears

                    bbox = detections[dis[0][0]].tlwh.copy()
                    bbox[2:] += bbox[:2]

                    check_if_append(detections, dis[i][0])
                    # bbox[1] -= (bbox[3]-bbox[1])/6
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1]+(bbox[3]-bbox[1])/6)), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                    # cv2.rectangle(img, (int(bbox[0]), int(bbox[1]), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                    if old_bbox is not None:
                        # cv2.rectangle(img, (int(old_bbox[0]), int(old_bbox[1])), (int(old_bbox[2]), int(old_bbox[3])), (255, 255, 255), 2)
                        pass
                    old_bbox = bbox.copy()
                    matched = True
                break

            if matched == False:
                print('lost ', time.time() - LAST_APPEAR, 's')
                if time.time() - LAST_APPEAR > 10:
                    managerStat = 'offline'
                else:
                    managerStat = 'leave'
                old_lost_time+=1
                if old_lost_time >= 5:
                    old_bbox = None
            else:
                LAST_APPEAR = time.time()
                managerStat = 'online'

                print('refound the target')
            if DEBUG_MODE:
                cv2.putText(img, managerStat, (50, 50), cv2.FONT_HERSHEY_PLAIN, fontScale=3, thickness=3, color=(0, 0, 255))

        else:
            for id, det in enumerate(detections):
                bbox = det.tlwh.copy()
                bbox[2:] += bbox[:2]
                # print(bbox)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)

        if not DEBUG_MODE:
            cv2.imwrite('/media/img/offline.jpg', img)
            publish.single('offlineImage',payload='x', hostname=MQTT_URL)
        if DEBUG_MODE:
            cv2.imshow('win', img)
            cv2.imwrite('/home/lyz/imgs/{:04d}.jpg'.format(idx), img)
            q = cv2.waitKey(30)
            if q == ord('q'):
                break
        idx += 1
        # pass

    if not DEBUG_MODE:
        cv2.destroyAllWindows()
        sub_thread.join()

if __name__ == '__main__':
    with open('../config/standing.json', 'r') as r:
        cfg = json.load(r)
    # print(cfg)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=os.path.abspath('/media/video/ch35_.mp4'))
    # parser.add_argument('--input', type=str, default=os.path.abspath('/media/video/ch74_2020-05-27-090034.mp4'))
    parser.add_argument('--cfg-pa', type=str, default=os.path.abspath('../config'))
    parser.add_argument('--use-model', type=bool, default=True)


    args = parser.parse_args()
    main(YOLO(), args, cfg)


