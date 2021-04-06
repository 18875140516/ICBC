#! /usr/bin/env python
# -*- coding: utf-8 -*-
#todo: test select_pattern is ok?
#todo: test config is ok？
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
import base64
from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from mgn.network import MGN
# from coordinate_transform import CoordTrans
from mgn.utils.extract_feature import extract_feature
from yolo import YOLO
from utils_icbc.util import checkPoint, compute_iou
import paho.mqtt.client as client
from configRetrive import ConfigRetrive
from rtmpAgent import RTMP_AGENT
from utils_icbc.util import checkPoint
import json
import logging
# from debug_cost_mat import *


warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
MQTT_URL = '211.67.21.65'
online_config = ConfigRetrive()
startTrack = False
tracked = False
clickPoint = None
WIDTH = None
HEIGHT = None
LAST_APPEAR = 0  # 记录上一次出现的时间
USE_MQTT = False
USE_IMSHOW = False
USE_FFMPEG = False
USE_INOTIFY = True

USE_PATTERN = True

agent = RTMP_AGENT(topic='offline', protocal='hls')
offline_client = client.Client()
offline_client.connect(host=MQTT_URL)
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
    cur_period = 0
    last_statistics_minute = 0
    hour_to_offline_time = dict()
    managerStat = 'online'
    # if USE_MQTT:
    #     def listenMQ():
    #     #从消息队列获得信息，然后改变当前状态
    #         def on_connect(client, userdata, flags, rc):
    #             print("Connected with result code " + str(rc))
    #
    #             # Subscribing in on_connect() means that if we lose the connection and
    #             # reconnect then subscriptions will be renewed.
    #             client.subscribe(topic='selectPerson')
    #             print('subscribe selectPerson successfully')
    #
    #         # The callback for when a PUBLISH message is received from the server.
    #         def on_message(client, userdata, msg):
    #             global startTrack
    #             global tracked
    #             global clickPoint
    #             global WIDTH
    #             global HEIGHT
    #             global LAST_APPEAR
    #             startTrack = True
    #             print('reveive msg', msg.payload)
    #             ret = json.loads(msg.payload)
    #             x = int(WIDTH*float(ret['x']))
    #             y = int(HEIGHT*float(ret['y']))
    #             print('x=',x, 'y=',y)
    #             clickPoint = (x, y)
    #             LAST_APPEAR = time.time()
    #             tracked = False
    #             print(msg.payload)
    #         client = mqtt.Client()
    #         client.on_connect = on_connect
    #         client.on_message = on_message
    #         print('starting connecting')
    #         ret = client.connect(MQTT_URL, 1883, 60)
    #         print(ret)
    #         print('ending connecting')
    #         client.loop_forever()
    #     sub_thread = threading.Thread(target=listenMQ)
    #     sub_thread.start()

    # Definition of the parameters
    max_cosine_distance = 0.3  # 允许相同人的最大余弦距离0.3
    nn_budget = None
    nms_max_overlap = 1.0  # 非极大值抑制，减少重复bbox，针对一类物体独立操作
    # deep_sort
    model_filename = 'model_data/mars-small128.pb'  # 128维特征预测模型，效果不佳，rank1极低
    # encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # mgn model
    print('loading mgn model')
    mgn = load_model_pytorch('/home/liuyongzhi/data/model.pt')
    print('load mgn OK')
    test_video_flag = True
    writeVideo_flag = False  # 是否写入视频

    if USE_IMSHOW:
        cv2.namedWindow('win')
        def clickCB(event, x, y, flag, param):

            if event == cv2.EVENT_LBUTTONDOWN:
                global startTrack
                global clickPoint
                global tracked
                clickPoint = (x, y)
                startTrack = True
                print(clickPoint)
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
        start = time.time()
        ret, img = cap.read()
        leaveTime = online_config.get('leaveTime', '999')
        leaveTime = int(leaveTime)
        if not ret:
            cap = cv2.VideoCapture(args.input)
            ret, img = cap.read()
        # img = cv2.resize(img,(720, 480))
        # print(img.shape)
        default_area = [[112, 271], [243, 488], [650, 431], [480, 230]]
        default_area = [[x[0]/img.shape[1], x[1]/img.shape[0]] for x in default_area]
        area = online_config.get('offline_area', default_area)
        area = [[int(x[0] * img.shape[1]), int(x[1] * img.shape[0])] for x in area]
        # img = cv2.resize(img, ( img.shape[1]//2,img.shape[0]//2))
        if ret == False:
            # print('read OK')
            cap = cv2.VideoCapture(args.input)
            ret, img = cap.read()
        boxs, classname = yolo.detect_image(Image.fromarray(img[...,::-1]))
        # print('yolov3 time(s): ', time.clock() - start)
        features = box_encode(mgn, Image.fromarray(img[..., ::-1]), boxs, prefix='./img1')
        # features = np.zeros(shape=(2048, 1))
        #todo: 特征提取后移,等选定了模板或选定的人后,在进行特征提取

        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # print('get detection time(s): ', time.clock() - start)

        #排除所选框外的detection
        det_out = []
        det_in = []
        for det in detections:
            tlbr = det.to_tlbr()
            if checkPoint(((tlbr[0] + tlbr[2]) // 2, tlbr[3]), area):
                det_in.append(det)
            else:
                det_out.append(det)
        for i in range(len(area)):
            cv2.line(img, tuple(area[i]), tuple(area[(i+1)%len(area)]),(0,0,255), thickness=2)
        detections = det_in


        if USE_PATTERN:
            #logging.info('get manager_pattern')
            manager_pattern = online_config.get('manager', None)# byte array
            # print("manager = ", manager_pattern)
            if manager_pattern is not None:
                tracked = True
                startTrack = True
                #assign target -> Detection
                img_base64decode = base64.b64decode(manager_pattern)
                img_array = np.frombuffer(img_base64decode, np.uint8)
                img_pattern = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
                fea = box_encode(mgn, Image.fromarray(img_pattern[..., ::-1]),
                                 [[0, 0, img_pattern.shape[1], img_pattern.shape[0]]])
                target = Detection([0,0,0,0], 1.0, fea)

        #点击了某个点后进行目标判定
        if tracked == False and startTrack == True: #choose detection
            print('1')
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
        #优选最相似目标
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

            #calculate offlineTime
            #todo: send offlinetime list to activemq
            #topic = 'config', {'offlineArray':[12 ,23, 12]}
            #test
            leaveTime = 1
            statistics_cycle = 1


            if matched and time.time() - LAST_APPEAR > leaveTime:
                cur_period += time.time() - LAST_APPEAR
                print('update cur period', cur_period)

            minute = time.asctime().split(':')[1]
            hour = time.asctime().split(':')[0].split(' ')[-1]
            if ( 60 + int(minute) - last_statistics_minute)%60 >= statistics_cycle:
                if hour in hour_to_offline_time.keys():
                    hour_to_offline_time[hour] += cur_period
                else:
                    hour_to_offline_time[hour] = cur_period
                #send to config
                offline_array = dict()
                offline_array['offlineArray'] = []
                print("hour_to_offline_time=", hour_to_offline_time)
                for i in range(8, 22):
                    if str(i) in hour_to_offline_time.keys():
                        offline_array['offlineArray'].append(int(hour_to_offline_time[str(i)]))
                print('offline_array', offline_array)
                last_statistics_minute = int(minute)
                offline_client.publish(topic='config', payload=json.dumps(offline_array))
            if int(hour) > 22:
                hour_to_offline_time.clear()



            if matched == False:
                # print('lost ', time.time() - LAST_APPEAR, 's')
                if time.time() - LAST_APPEAR > leaveTime:
                    managerStat = 'offline'
                else:
                    managerStat = 'leave'
                old_lost_time+=1
                if old_lost_time >= 5:
                    old_bbox = None
            else:
                LAST_APPEAR = time.time()
                managerStat = 'online'

            cv2.putText(img, managerStat, (50, 50), cv2.FONT_HERSHEY_PLAIN, fontScale=3, thickness=3, color=(0, 0, 255))

            root = dict()
            root['status'] = managerStat
            s = json.dumps(root)
            offline_client.publish(topic='managerStatus', payload=s)

        #选定前的画面
        else:
            for id, det in enumerate(detections):
                bbox = det.tlwh.copy()
                bbox[2:] += bbox[:2]
                # print(bbox)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            root = dict()
            root['status'] = 'default'
            s = json.dumps(root)
            offline_client.publish(topic='managerStatus', payload=s)

        if USE_IMSHOW:#所有分支都需要
            cv2.imshow('win', img)
            cv2.imwrite('/home/lyz/imgs/{:04d}.jpg'.format(idx), img)
            q = cv2.waitKey(30)
            if q == ord('q'):
                break
        elif USE_MQTT:
            cv2.imwrite('/media/img/after.jpg', img)
            s = base64.b64encode(cv2.imencode('.jpg', img)[1])
            offline_client.publish('offlineImage',payload=s)
        elif USE_FFMPEG:
            agent.send_image(img)
            pass
        elif USE_INOTIFY:
            cv2.imwrite('/tmp/flow_offline.jpg', img)
        print('frame: ', idx, 'cost time(ms):', str(1000*(time.time() - start)).split('.')[0])
        idx += 1

    if USE_IMSHOW:
        cv2.destroyAllWindows()
        # sub_thread.join()

if __name__ == '__main__':
    with open('../config/standing.json', 'r') as r:
        cfg = json.load(r)
    # print(cfg)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=os.path.abspath('/media/video/test.avi'))
    # parser.add_argument('--input', type=str, default=os.path.abspath('/media/video/ch74_2020-05-27-090034.mp4'))
    parser.add_argument('--cfg-pa', type=str, default=os.path.abspath('../config'))
    parser.add_argument('--use-model', type=bool, default=True)


    args = parser.parse_args()
    main(YOLO(), args, cfg)


