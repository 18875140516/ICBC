# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
# from debug_cost_mat import *

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """
    _next_id = 1
    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3,usemodel="mgn"):#default 3
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.use_model = usemodel
        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        # self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.级连匹配
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
	# print(matches)
        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])#如果发现新的detection，那么就initiate一个

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)#将没有被deleted的track，的特征以及对应的id，当前confirmed id输入
        return matches, unmatched_detections

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)#输入detection，tracker_ids,获得代价矩阵

            #采用匈牙利算法将距离矩阵变成assign矩阵，每一个tracker只能和一个det匹配
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            # print('from match\n', cost_matrix)
            return cost_matrix
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        # print('-inner euc, use comfirmed')
        # print('--detections', detections)
        # print('--confirmed_tracks',confirmed_tracks)
        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks, debug='inner')
        # print('--matches_a',matches_a)
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]#track需要达到连续三帧才会变成状态confirmed,

        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]


        # print('-inner iou')
        # print('--iou_track_candidates',iou_track_candidates)
        # print('--unmatched_detections',unmatched_detections)
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections, debug='iou')
        # print('--iou_match', matches_b)
        matches = matches_a + matches_b
        # print('-match', matches)
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        # print('-unmatched_tracks', unmatched_tracks)
        return matches, unmatched_tracks, unmatched_detections

    #创建一个新的track
    def _initiate_track(self, detection, id=-1):
        if id == -1:
            mean, covariance = self.kf.initiate(detection.to_xyah())
            self.tracks.append(Track(
                mean, covariance, self._next_id, self.n_init, self.max_age,
                detection.feature))
            # print('get a new track', self._next_id)
            Tracker._next_id += 1
        else:
            mean, covariance = self.kf.initiate(detection.to_xyah())
            self.tracks.append(Track(
                mean, covariance, id, self.n_init, self.max_age,
                detection.feature))
            # print('get a old track', id)

    def updateInAllTracker(self, detections, trackers):
        # print('updateInAllTracker inner match result')
        # Run matching cascade.级连匹配
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        # print('detection', len(detections))


        exist_ids = [track.track_id for track in self.tracks if track.is_confirmed]
        # get other trackers
        trackers = [tracker for tracker in trackers if tracker != self]
        #与其他tracker对象所维护的track目标 匹配
        for _tracker in trackers:
            def gated_metric(tracks, dets, track_indices, detection_indices):
                features = np.array([dets[i].feature for i in detection_indices])
                targets = np.array([tracks[i].track_id for i in track_indices])
                # 通过距离函数直接求解detection之间的距离，不适用卡尔曼滤波进行判断
                cost_matrix = _tracker.metric.distance(features, targets)  # 输入detection，tracker_ids,获得代价矩阵

                # 采用匈牙利算法将距离矩阵变成assign矩阵，每一个tracker只能和一个det匹配
                # cost_matrix = linear_assignment.gate_cost_matrix(
                #     self.kf, cost_matrix, tracks, dets, track_indices,
                #     detection_indices)
                # print('det', detection_indices)
                # print('target', targets)
                # print('from others', cost_matrix)
                return cost_matrix

            # confirmed_tracks = [
            #     i for i, t in enumerate(_tracker.tracks) if t.is_confirmed()]#the confirmed track in the tracker
            confirmed_tracks = [
                i for i, t in enumerate(_tracker.tracks) if t.is_confirmed() and t.track_id not in exist_ids]

            # confirmed_tracks = [t.track_id for t in _tracker.tracks if t.is_confirmed() and t.track_id  not in exist_ids]
            # use the metric of _tracker to match these res_detection
            # matches_b, unmatched_tracks_b, unmatched_detections_b = \
            #     linear_assignment.matching_cascade(
            #         gated_metric, _tracker.metric.matching_threshold, _tracker.max_age,
            #         _tracker.tracks, detections, confirmed_tracks, unmatched_detections)#todo: 是否是confirm

            matches_b, unmatched_tracks_b, unmatched_detections_b = \
                linear_assignment.matching_cascade(
                    gated_metric, self.metric.matching_threshold, _tracker.max_age,
                    _tracker.tracks, detections, confirmed_tracks, unmatched_detections, debug='inter')#todo: 是否是confirm

            # print('updateInAllTracker inter match result')


            #need to be init
            matches_b = [(_tracker.tracks[a].track_id, b) for a, b in matches_b]#track_id and idx of detections
            # print(matches_b)
            #将找到满足距离要求的det剔除
            unmatched_detections = [i for i in unmatched_detections if i not in [j for i, j in matches_b]]
            # print('匹配到',matches_b, '未匹配', unmatched_detections)
            for id, detection_idx in matches_b:
                # print('find a old id from other camera:', id)
                self._initiate_track(detections[detection_idx], id)


        #仅更新了第一次匹配到的
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        margin = 30
        #todo: 如果未到边界，那么不删除track！！
        for track_idx in unmatched_tracks:
            # self.time_since_update = 0  # 此次进行了更新，所以该参数为0
            # if self.tracks[track_idx].state == 1 and self.tracks[track_idx].hits >= self.tracks[track_idx]._n_init:
            #     self.tracks[track_idx].state = 2
            # pass
            # bbox =self.tracks[track_idx].to_tlwh
            # if bbox[0] < margin || bbox[1] < margin || bbox[0] + bbox[1] >
            self.tracks[track_idx].mark_missed()

        #deault, init all of the unmatched det even it matched in other tracker
        #初始化所有未匹配到的det，未与新的tracker匹配
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])#如果发现新的detection，那么就initiate一个
            # print('find a new id:', self._next_id-1)

        #将被deleted的track剔除
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []#features为track的元素，每一个track包含多个feature，将其连续排列， targets为feature对应的id
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)#将没有被deleted的track，的特征以及对应的id，当前confirmed id输入
        return matches, unmatched_detections
