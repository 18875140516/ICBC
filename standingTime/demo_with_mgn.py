#demo_with_mgn.py
#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO, args

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from tools.generate_detections import tracklet
from deep_sort.detection import Detection as ddet
# from coordinate_transform import CoordTrans
from mgn.utils.extract_feature import extract_feature
from mgn.network import MGN
from torchvision.datasets.folder import default_loader
import torch
from tqdm import tqdm
import matplotlib
from torchvision import transforms
import os

import time
from opt import args
# from debug_cost_mat import *
from opt import args
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

    img.save(os.path.join(prefix, 'raw.jpg'), quality=95)
    for idx, x in enumerate(imgs):
        x.save(os.path.join(prefix, 'box'+str(idx)+'.jpg'), quality=95)

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


def main(yolo, videoPath=''):  # 输入yolov3模型和视频路径

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

    # if videoPath != '':
    #     video_capture = cv2.VideoCapture(videoPath)
    # else:
    #     video_capture = cv2.VideoCapture(0)

    fps = 0.0

    cap1 = cv2.VideoCapture(args['input1'])
    cap2 = cv2.VideoCapture(args['input2'])
    k = 1
    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(cap1.get(3))
        h = int(cap1.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w , h*2))
        list_file = open('detection.txt', 'w')
        frame_index = -1
    idx = 1
    caps = [cap1, cap2]
    trackers = []
    tracklets = []
    for i in range(len(caps)):
        metric = nn_matching.NearestNeighborDistanceMetric("mgn", 0.3, nn_budget)
        trackers.append(Tracker(metric, usemodel='mgn'))
        tmp = tracklet()
        tracklets.append(tmp)

    if not os.path.exists('./img1'):
        os.mkdir('./img1')
    if not os.path.exists('./img2'):
        os.mkdir('./img2')
    CNT = len(trackers)

    while True:
        # tmp = 1
        # while tmp > 0:
        #     tmp-=1
        #     for cap in caps:
        #         ret, frame = cap.read()
        # print('frame', idx // 2, '-------------------------------------------')
        t1 = time.time()
        if writeVideo_flag:
            out_img = []
        old = []
        for cap in caps:
            idx += 1
            print()
            print('---------------------------------------------')
            print('frame = ', idx)

            tracker = trackers[idx % CNT]
            _tracklet = tracklets[idx%CNT]
            # print("len of tracks", len(tracker.tracks))
            ret, frame = cap.read()


            if writeVideo_flag:
                out_img.append(frame)
            if ret != True:
                break

            if test_video_flag and (idx == 3 or idx == 5):
                continue

            # image = Image.fromarray(frame)
            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
            boxs, classname = yolo.detect_image(image)#xy w h
            # print("box_num", len(boxs))
            # features = encoder(frame, boxs)  # 输入一张图片和多个bbox，[bbox1, bbox2, bbox3,...]
            if idx %2 ==1 and not os.path.exists('./img1/frame' + str(idx)):
                os.mkdir('./img1/frame' + str(idx))
            elif idx % 2== 0 and not os.path.exists('./img2/frame' + str(idx)):
                os.mkdir('./img2/frame' + str(idx))

            if args["record"] == "1":
                for index, _box in enumerate(boxs):
                    _box[2] += _box[0]
                    _box[3] += _box[1]
                    person = frame[ _box[1]:_box[3],_box[0]: _box[2]]
                    if idx %2 ==1:
                        cv2.imwrite('./img1/frame'+str(idx) + '/' + str(index)+'.jpg', person)
                    else:
                        cv2.imwrite('./img2/frame' + str(idx) + '/' + str(index) + '.jpg', person)
                    _box[2] -= _box[0]
                    _box[3] -= _box[1]

            if idx%2 == 1:
                features = box_encode(mgn, Image.fromarray(frame[..., ::-1]), boxs, prefix='./img1')
            else:
                features = box_encode(mgn, Image.fromarray(frame[..., ::-1]), boxs, prefix='./img2')






            # score to 1.0 here).
            # 每一对bbox与feature生成一个detection
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            det_frame = frame.copy()
            for id, det in enumerate(detections):
                bbox = det.tlwh.copy()
                bbox[2:] += bbox[:2]
                print(bbox)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                cv2.putText(frame, str(id+1),(int(bbox[0]), int(bbox[1]+20)),0, 5e-3 * 200, (0,255,0),1)
            # cv2.imshow('det', det_frame)
            # Call the tracker
            tracker.predict()

            # matches, unmatch_det = tracker.update(detections)#利用detection更新tracker
            matches, unmatch_det= tracker.updateInAllTracker(detections, trackers)  # 利用detection更新tracker
            ids = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:#这个地方表示最近两帧中的对象都显示出来
                    continue
                color = COLORS[track.track_id % 200]
                ids.append(track.track_id)
                bbox = track.to_tlbr()
                _tracklet.setpoint(track.track_id, (int(bbox[0] + bbox[2]) // 2, int(bbox[1] + bbox[3]) // 2))
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

                # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), tuple(color), 2)
                cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
                print('track len =  ',len(track.features))
                # g_gallery.set_feature(name=track.track_id, feature=track.features[-1], frame=idx)

            cv2.putText(frame, str(idx), (50,50), 0, 5e-3 * 200, (0, 255, 0), 2)
            frame = _tracklet.draw_tracklet(frame, ids)
            cv2.imshow(str(idx % len(caps)), frame)
            old.append(frame)



        fps = (fps + (1. / (time.time() - t1))) / 2
        cat_img = np.concatenate(old, axis=0)
        # cv2.imshow('cat', cat_img)
        cv2.imwrite('./old/frame{}.jpg'.format(idx), cat_img)
        # print("fps= %f"%(fps))
        if args['is_debug'] == "1":
            if cv2.waitKey(0) & 0xFFFF == ord('q'):
                break
        else:
            if cv2.waitKey(1) & 0xFFFF == ord('q'):
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
    main(YOLO())
    # main(YOLO())

