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
from utils_icbc.util import checkPoint
# from debug_cost_mat import *
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

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

    # Definition of the parameters
    max_cosine_distance = 0.3  # 允许相同人的最大余弦距离0.3
    nn_budget = None
    nms_max_overlap = 1.0  # 非极大值抑制，减少重复bbox，针对一类物体独立操作
    # deep_sort
    model_filename = 'weigths/mars-small128.pb'  # 128维特征预测模型，效果不佳，rank1极低
    # encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # mgn model
    print('loading mgn model')
    mgn = load_model_pytorch('/home/lyz/Desktop/ReID-MGN/model.pt')
    print('load mgn OK')
    test_video_flag = True
    writeVideo_flag = False  # 是否写入视频


    fps = 0.0

    cap = cv2.VideoCapture(args.input)
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

    debug_move = np.zeros(shape=(3, 3), dtype=np.int)
    area_hash = dict()
    for idd, area in enumerate(cfg['areaNames']):
            area_hash[area] = idd


    while True:
        t1 = time.time()
        if writeVideo_flag:
            out_img = []
        ret, frame = cap.read()
        if ret == False:
            break
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, classname = yolo.detect_image(image)  # xy w h
        features = box_encode(mgn, Image.fromarray(frame[..., ::-1]), boxs, prefix='./img1')
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]


        #draw area
        # for i in range(len(cfg['area'])):
        #     cv2.line(frame, tuple(cfg['area'][i]), tuple(cfg['area'][(i+1)%len(cfg['area'])]),(0,0,255), thickness=2)
        for area in cfg['areaNames']:
            pt = [0, 0]
            for i in range(len(cfg[area])):
                pt[0] += cfg[area][i][0]
                pt[1] += cfg[area][i][1]
                cv2.line(frame, tuple(cfg[area][i]), tuple(cfg[area][(i + 1) % len(cfg[area])]), (0, 0, 255),
                         thickness=2)
            pt = [i//len(cfg[area]) for i in pt]
            cv2.putText(frame, area, tuple(pt), cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,0,255), thickness=2 )

        det_frame = frame.copy()
        # for id, det in enumerate(detections):
        #     bbox = det.tlwh.copy()
        #     bbox[2:] += bbox[:2]
        #     print(bbox)
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)


        # cv2.imshow('det', det_frame)
        # Call the tracker
        tracker.predict()
        matches, unmatch_det = tracker.update(detections)  # 利用detection更新tracker
        #update tracks status
        for track in tracker.tracks:
            tlbr = track.to_tlbr()
            curArea = 'outside'
            for area in cfg['areaNames']:
                if checkPoint(((tlbr[0]+tlbr[2])//2, tlbr[3]),cfg[area]):
                    curArea = area
                    break
            ret = track.update_area(curArea)
            if ret is not None:
                ti, fr, to = ret
                print('track: ', track.track_id, "from",fr+'->'+to)
                debug_move[area_hash[fr]][area_hash[to]] += 1
                # cv2.putText(frame, fr+'->'+to, (50,50), cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,0,255))
            # track.update_time(checkPoint(((tlbr[0]+tlbr[2])//2, tlbr[3]),cfg['area']))

        hh = 20
        for fr in cfg['areaNames']:
            for to in cfg['areaNames']:
                if fr != to:
                    cv2.putText(frame, fr+'->'+to+':'+str(debug_move[area_hash[fr]][area_hash[to]]),
                                (50, hh), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), thickness=2)
                    hh += 25

        # cv2.putText(frame, 'area 1->2 : ' + str(debug_move[area_hash['area1']['area2']]) ,)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:  # 这个地方表示最近两帧中的对象都显示出来
                continue
            color = COLORS[track.track_id % 200]
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            # cv2.putText(frame, str(track.standing_time()) + 's', (int(bbox[0]), int(bbox[1])-20), 0, 5e-3 * 200, (0, 255, 0), 4)

            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.imshow('win', frame)
        cv2.imwrite('/media/lyz/9E3B-0828/12-18/{:06d}.jpg'.format(idx),frame)
        fps = (fps + (1. / (time.time() - t1))) / 2
        idx += 1
        if cv2.waitKey(30) == ord('q'):
            break


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

    # video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    with open('../config/crossRegion.json', 'r') as r:
        cfg = json.load(r)
    print(cfg)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/media/video/test.avi')
    parser.add_argument('--cfg-pa', type=str, default='../config')


    args = parser.parse_args()
    main(YOLO(), args, cfg)


