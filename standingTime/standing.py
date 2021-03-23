#! /usr/bin/env python
# -*- coding: utf-8 -*-
#ok: add queue model to standing time
#todo: add wait number/time to mysql
'''
wait number = average number of one minute
wait time = acerage time of one minute
write to mysql when current minute change
old_min record old minute
current time = time.asctime().split(':')[1]
'''
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
from configRetrive import ConfigRetrive
import base64
import logging
from rtmpAgent import RTMP_AGENT
from udn_socket import UDNClient
import paho.mqtt.publish as publish
import pymysql
# from debug_cost_mat import *
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
online_config = ConfigRetrive()
USE_IMSHOW = False
USE_MQTT = False
USE_FFMPEG = False
USE_INOTIFY = True
USE_UDN = False

udn_client = UDNClient('standing')
MQTT_URL = online_config.get('MQTT_URL', '127.0.0.1')
RTMP_URL = online_config.get("RTMP_URL", "127.0.0.1")
RTMP_PORT = online_config.get('RTMP_PORT', 1935)
MYSQL_URL = online_config.get('MYSQL_URL', '127.0.0.1')
MYSQL_PORT = online_config.get('MYSQL_PORT', 10086)
standing_db = pymysql.connect(user='lyz', password='lyz', host=MYSQL_URL,port=MYSQL_PORT,database='mysql')
# test_cursor = standing_db.cursor()

STANDING_TOPIC = online_config.get('STANDING_TOPIC', "standing")

standing_agent = RTMP_AGENT(rtmp_url=RTMP_URL, rtmp_port=RTMP_PORT, topic=STANDING_TOPIC)
def load_model_pytorch(model_path):
    logging.info('loading model pytorch')
    model = MGN()
    model = model.to('cuda')
    model.load_state_dict(torch.load(model_path))
    logging.info('load ok')
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


def main(yolo, args):  # 输入yolov3模型和视频路径
    logging.info('standing main')
    old_time = time.asctime().split(':')[1]
    sum_time = 0
    sum_number = 0
    frequency = 0

    # Definition of the parameters
    max_cosine_distance = 0.3  # 允许相同人的最大余弦距离0.3
    nn_budget = None
    nms_max_overlap = 1.0  # 非极大值抑制，减少重复bbox，针对一类物体独立操作
    # deep_sort
    model_filename = 'model_data/mars-small128.pb'  # 128维特征预测模型，效果不佳，rank1极低
    # encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # mgn model
    mgn = load_model_pytorch('/home/lyz/Desktop/ReID-MGN/model.pt')
    test_video_flag = True
    writeVideo_flag = False  # 是否写入视频


    fps = 0.0
    cap = cv2.VideoCapture(args.input)
    #send to config
    ret, img = cap.read()
    assert ret == True
    # img = cv2.resize(img, (640, 480))
    # img_b64 = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
    # topic = 'standing_area'
    # publish.single(topic='config',payload=json.dumps({topic:img_b64}),hostname=MQTT_URL)


    k = 1
    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(cap.get(3))
        h = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w , h*2))
        list_file = open('detection.txt', 'w')
        frame_index = -1
    idx = 1
    metric = nn_matching.NearestNeighborDistanceMetric("mgn", 0.3, nn_budget)
    tracker = Tracker(metric, usemodel='mgn')

    while True:
        frequency += 1


        t1 = time.time()
        if writeVideo_flag:
            out_img = []
        ret, frame = cap.read()

        if ret == False:
            cap = cv2.VideoCapture(args.input)
            ret, frame = cap.read()
        default_area = [[112, 271], [243, 488], [650, 431], [480, 230]]
        default_area = [[x[0]/frame.shape[1], x[1]/frame.shape[0]] for x in default_area]
        area = online_config.get('standing_area', default_area)
        if type(area) != list or len(area) == 0:
            continue
        if max(max(area)) > 1:
            print('area is not valid')
            continue
        area = [[int(x[0]*frame.shape[1]), int(x[1]*frame.shape[0])] for x in area]
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, classname = yolo.detect_image(image)  # xy w h
        features = box_encode(mgn, Image.fromarray(frame[..., ::-1]), boxs, prefix='./img1')
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        det_out = []
        det_in = []
        for det in detections:
            tlbr = det.to_tlbr()
            if checkPoint(((tlbr[0] + tlbr[2])//2, tlbr[3]), area):
                det_in.append(det)
            else:
                det_out.append(det)

        queueSize = len(det_in)

        #draw area
        if online_config.get('SHOW_AREA', False):
            for i in range(len(area)):
                cv2.line(frame, tuple(area[i]), tuple(area[(i+1)%len(area)]),(0,0,255), thickness=2)
        cv2.putText(frame, 'queue: ' + str(queueSize), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2)
        det_frame = frame.copy()
        if online_config.get('SHOW_BBOX', False):
            for id, det in enumerate(det_out):
                bbox = det.tlwh.copy()
                bbox[2:] += bbox[:2]
                # print(bbox)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)

        detections = det_in
        tracker.predict()
        matches, unmatch_det = tracker.update(detections)  # 利用detection更新tracker
        #update tracks status
        for track in tracker.tracks:
            tlbr = track.to_tlbr()
            track.update_time(checkPoint(((tlbr[0]+tlbr[2])//2, tlbr[3]),area))

        mostStaningTime = 0
        sum_time_item = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:  # 这个地方表示最近两帧中的对象都显示出来
                continue
            color = COLORS[track.track_id % 200]
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, str(track.standing_time()) + 's', (int(bbox[0]), int(bbox[1])-20), 0, 5e-3 * 200, (0, 255, 0), 2)
            sum_time_item.append(track.standing_time())
            mostStaningTime = max(mostStaningTime, track.standing_time())
            # cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        # print(sum_time_item)
        sum_time += sum(sum_time_item)//max(len(sum_time_item),1)
        # if args.use_model == True and idx % 5 == 1:
        #     cv2.imwrite('/media/image/head.jpg', frame)
            # publish.single('image', payload='c', hostname='127.0.0.1')
        #transfer to minute
        mostStaningTime = mostStaningTime//60

        #handle mqtt message
        if queueSize > int(online_config.get('waitNumber', 999)):
            cv2.putText(frame, 'too much people', (200, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                        thickness=2)
            root = dict()
            root['message'] = 'too much people'
            publish.single('warning', payload=json.dumps(root), hostname=MQTT_URL)
        if mostStaningTime > int(online_config.get('waitTime', '999')):
            cv2.putText(frame, 'wait too long', (200, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                        thickness=2)
            root = dict()
            root['message'] = 'customer wait too long'
            publish.single('warning', payload=json.dumps(root), hostname=MQTT_URL)
            pass

        sum_number += queueSize

        #check new data, write to mysql
        if time.asctime().split(':')[1] != old_time:
            cursor = standing_db.cursor()
            print("insert into wait_infos(timestamp, wait_time, wait_number) values('{}', '{}', '{}')"
                           .format(time.asctime(), sum_time//frequency, sum_number//max(frequency, 1)))
            cursor.execute("insert into wait_infos(timestamp, wait_time, wait_number) values('{}', '{}', '{}')"
                           .format(time.asctime(), sum_time//frequency, sum_number//max(frequency, 1)))
            standing_db.commit()
            frequency  = 0
            sum_number = 0
            sum_time = 0
            old_time = time.asctime().split(':')[1]


        num_json = dict()
        standing_json = dict()
        num_json['numberOfQueue'] = queueSize
        publish.single('numQueue', payload=json.dumps(num_json), hostname=MQTT_URL)

        standing_json['mostStandingTime'] = str(mostStaningTime)
        # publish.single('mostStandingTime', payload=json.dumps(num_json), hostname=MQTT_URL)
        publish.single('mostStandingTime', payload=json.dumps(standing_json), hostname=MQTT_URL)

        cv2.putText(frame, time.asctime(), (50, 100), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 255, 255))

        if USE_IMSHOW:
            cv2.putText(frame, time.asctime(),(50,100), cv2.FONT_HERSHEY_PLAIN, fontScale=1,color=(255,255,255))
            cv2.imshow('win', frame)
            q = cv2.waitKey(30)
            if q == ord('q'):
                break
            elif q == ord('x'):
                cv2.imwrite('{:05d}.jpg'.format(idx), frame)
        elif USE_MQTT:
            s = base64.b64encode(cv2.imencode('.jpg', frame)[1])
            publish.single(topic='offlineImage', hostname=MQTT_URL, payload=s)
        elif USE_FFMPEG:
            standing_agent.send_image(frame)
        elif USE_UDN:
            udn_client.send_img(frame)
            pass
        elif USE_INOTIFY:
            cv2.imwrite('/tmp/standing.jpg', frame)
        fps = (time.time() - t1)*1000
        print(fps)
        idx += 1

        # if writeVideo_flag:
        # #     # save a frame
        #     out_img = np.array(out_img)
        #     out_img = np.concatenate([out_img[0], out_img[1]], 0)
        #     # save a frame
        #     out.write(out_img)

        #     frame_index = frame_index + 1
        #     list_file.write(str(frame_index)+' ')
        #     if len(boxs) != 0:
        #         for i in range(0,len(boxs)):
        #             list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
        #     list_file.write('\n')
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    video_path = online_config.get('VIDEO_PATH', '/media/video/ch35_.mp4')
    #video_path = '/media/video/ch35_.mp4'
    parser.add_argument('--input', type=str, default=os.path.abspath(video_path))
    parser.add_argument('--use-model', type=bool, default=True)


    args = parser.parse_args()
    main(YOLO(), args)


