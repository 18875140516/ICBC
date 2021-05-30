#! /usr/bin/env python
# -*- coding: utf-8 -*-
#todo: test select_pattern is ok?
#todo: test config is ok？
from __future__ import division, print_function, absolute_import
import sys
sys.path.append('../')
import argparse
import json
import os
import time
import warnings
import sys
sys.path.append(os.path.abspath(os.getcwd()))
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
#from yolo import YOLO
from yolov5.simple_detect import simple_yolov5 as YOLO
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
MQTT_URL = '211.67.22.33'
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
root = dict()
root['hour_to_offline_time'] = dict()
offline_client.publish(topic='config', payload=json.dumps(root))

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
    LAST_APPEAR = time.time()
    cur_period = 0
    last_statistics_minute = -10

    
    hour_to_offline_time = dict()
    hour_to_offline_count = dict()
    hour_to_offline_time = online_config.get('hour_to_offline_time', {'12':12}) 
    hour_to_offline_count = online_config.get('hour_to_offline_count', {})

    COUNTED = False
    managerStat = 'online'
    max_cosine_distance = 0.3  # 允许相同人的最大余弦距离0.3
    nn_budget = None
    nms_max_overlap = 1.0  # 非极大值抑制，减少重复bbox，针对一类物体独立操作
    # deep_sort
    model_filename = 'weights/mars-small128.pb'  # 128维特征预测模型，效果不佳，rank1极低
    # encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # mgn model
    print('loading mgn model')
    mgn = load_model_pytorch(args.weights)
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
    statistics_hour = time.asctime().split(':')[0].split(' ')[-1]#
    cycle_send_to_config_min = 5#min
    cycle_update_s = 10#s
    launch_time = time.time()
    while True:
        start = time.time()
        minute = time.asctime().split(':')[1]
        hour = str(int(time.asctime().split(':')[0].split(' ')[-1]))
        ret, img = cap.read()
        leaveTime = online_config.get('leaveTime', '999')
        leaveTime = int(leaveTime)
        leaveTime = 10#当超过leaveTime时间后则判断为offline
        statistics_cycle = 1
        if not ret:
            cap = cv2.VideoCapture(args.input)
            ret, img = cap.read()
        # img = cv2.resize(img,(720, 480))
        default_area = [[112, 271], [243, 488], [650, 431], [480, 230]]
        default_area = [[x[0]/img.shape[1], x[1]/img.shape[0]] for x in default_area]
        area = online_config.get('offline_area', default_area)
        area = [[int(x[0] * img.shape[1]), int(x[1] * img.shape[0])] for x in area]
        # img = cv2.resize(img, ( img.shape[1]//2,img.shape[0]//2))
        if ret == False:
            cap = cv2.VideoCapture(args.input)
            ret, img = cap.read()
        Image_img = Image.fromarray(img[...,::-1])
        boxs, classname = yolo.detect_image(img)
        features = box_encode(mgn, Image.fromarray(img[..., ::-1]), boxs, prefix='./img1')
        # features = np.zeros(shape=(2048, 1))
        #todo: 特征提取后移,等选定了模板或选定的人后,在进行特征提取

        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

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
            for det in detections:
                bbox = det.tlwh.copy()
                bbox[2:] += bbox[:2]
                records.append(det.feature)
                if clickPoint[0] > bbox[0] \
                    and clickPoint[0] < bbox[2]\
                    and clickPoint[1] > bbox[1]\
                    and clickPoint[1] < bbox[3]:
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
                # cv2.putText(img, str(dis[i][0]), (int(cmp_bbox[1]), int(cmp_bbox[0])), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
                def check_if_append(detections, check_id):
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

                    pass

                # if old_bbox is not None and compute_iou(old_bbox, cmp_bbox) < 0.01:
                #     continue

                if dis[i][1] >= 0.8:

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

            tt = time.time()
            if tt - LAST_APPEAR > leaveTime and not matched:#offline too long
                if statistics_hour in hour_to_offline_count.keys():
                    hour_to_offline_count[statistics_hour] += 1
                else:
                    hour_to_offline_count[statistics_hour] = 1
                managerStat = 'offline'
                cur_period += tt - LAST_APPEAR
                LAST_APPEAR = tt
            elif tt - LAST_APPEAR > leaveTime and matched:#
                managerStat = 'online'
                cur_period += tt - LAST_APPEAR
                LAST_APPEAR = tt
            elif tt - LAST_APPEAR < leaveTime and matched and managerStat == 'leave':#leave to online
                managerStat = 'online'
                LAST_APPEAR = tt
            elif tt - LAST_APPEAR < leaveTime and matched and managerStat == 'offline':#offline to online
                cur_period += tt - LAST_APPEAR
                managerStat = 'online'
                LAST_APPEAR = tt
            elif tt - LAST_APPEAR < leaveTime and matched and managerStat == 'online':
                LAST_APPEAR = tt
            elif tt - LAST_APPEAR < leaveTime and not matched:
                if managerStat == 'online' or managerStat == 'leave':
                    managerStat = 'leave'
                elif managerStat == 'offline':
                    managerStat = 'offline'
            # cv2.putText(img, str(cur_period).split('.')[0], (50, 100), cv2.FONT_HERSHEY_PLAIN, fontScale=3, thickness=3, color=(0, 0, 255))
            # cv2.putText(img, str(LAST_APPEAR).split('.')[0], (50, 150), cv2.FONT_HERSHEY_PLAIN, fontScale=3, thickness=3, color=(0, 0, 255))
            # cv2.putText(img, str(tt - launch_time).split('.')[0], (50, 200), cv2.FONT_HERSHEY_PLAIN, fontScale=3, thickness=3,
            #             color=(0, 0, 255))
            #跨越小时，更新statistics_hour

            # cv2.putText(img, managerStat, (50,50), cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(255,255,255))
            print(managerStat)

            if ( 60 + int(minute) - last_statistics_minute)%60 >= statistics_cycle:
                hour_to_offline_time[statistics_hour] = cur_period
                if hour != statistics_hour:
                    statistics_hour = hour
                    cur_period = 0
                last_statistics_minute = int(minute)

                root = dict()
                root['hour_to_offline_time'] = hour_to_offline_time
                offline_client.publish(topic='config', payload=json.dumps(root))

                root = dict()
                root['hour_to_offline_count'] = hour_to_offline_count
                offline_client.publish(topic='config', payload=json.dumps(root))
                print("publish to config=",hour_to_offline_count,hour_to_offline_time)
            if int(hour) > 22:
                #12点过后，离岗数据清除
                hour_to_offline_time.clear()
                hour_to_offline_count.clear()
                root = dict()
                root['hour_to_offline_time'] = hour_to_offline_time

                offline_client.publish(topic='config', payload=json.dumps(root))
                root = dict()
                root['hour_to_offline_count'] = hour_to_offline_count

                offline_client.publish(topic='config', payload=json.dumps(root))


            #
            # if matched == False:
            #     # print('lost ', time.time() - LAST_APPEAR, 's')
            #     if time.time() - LAST_APPEAR > leaveTime:
            #         managerStat = 'offline'
            #         if not COUNTED:
            #             if hour not in hour_to_offline_count.keys():
            #
            #                 hour_to_offline_count[hour] = 1
            #             else:
            #                 hour_to_offline_count[hour] += 1
            #                 print('offline count ++')
            #             #offline_count_array = dict()
            #             #offline_count_array['offlineCountArray'] = []
            #             #for i in range(9, int(hour)+1):
            #             #    if str(i) in hour_to_offline_count.keys():
            #             #        offline_count_array['offlineCountArray'].append(int(hour_to_offline_count[str(i)]))
            #             #    else:
            #             #        offline_count_array['offlineCountArray'].append(0)
            #
            #             root = dict()
            #             root['hour_to_offline_count'] = hour_to_offline_count
            #             offline_client.publish(topic='config', payload=json.dumps(root))
            #             COUNTED = True
            #     else:
            #         managerStat = 'leave'
            #     old_lost_time+=1
            #     if old_lost_time >= 5:
            #         old_bbox = None
            # else:
            #     COUNTED = False
            #     LAST_APPEAR = time.time()
            #     managerStat = 'online'


            root = dict()
            status_cn = ''
            if managerStat == 'online':
                status_cn = '在线'
            elif managerStat == 'offline':
                status_cn = '离岗'
            elif managerStat == 'leave':
                status_cn = '暂离'
            root['status'] = status_cn
            s = json.dumps(root)
            offline_client.publish(topic='managerStatus', payload=s)

        #选定前的画面
        else:
            for id, det in enumerate(detections):
                bbox = det.tlwh.copy()
                bbox[2:] += bbox[:2]
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
            # cv2.imshow('img', img)
            # cv2.waitKey(30)
            cv2.imwrite('/tmp/flow_offline.jpg', img)
        idx += 1

    if USE_IMSHOW:
        cv2.destroyAllWindows()
        # sub_thread.join()

if __name__ == '__main__':
    with open('../config/standing.json', 'r') as r:
        cfg = json.load(r)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=os.path.abspath('/media/video/test.avi'))
    # parser.add_argument('--input', type=str, default=os.path.abspath('/media/video/ch74_2020-05-27-090034.mp4'))
    parser.add_argument('--cfg-pa', type=str, default=os.path.abspath('../config'))
    parser.add_argument('--use-model', type=bool, default=True)
    parser.add_argument('--weights', type=str, default='/home/liuyongzhi/data/model.pt')
    parser.add_argument('--weights_yolov5', type=str, default='/home/liuyongzhi/data/yolov5x.pt')



    args = parser.parse_args()
    main(YOLO(weights=args.weights_yolov5), args, cfg)


