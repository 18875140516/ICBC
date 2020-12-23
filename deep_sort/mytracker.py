# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
# from .track import Track
from tracker.gallery import g_gallery
from deep_sort.detection import Detection
from deep_sort.basetrack import BaseTrack
import lap

class myTrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class myTrack(BaseTrack):
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """
    next_id = 1
    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None, alpha=0.1):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.alpha = alpha
        self.time_since_update = 0

        self.state = myTrackState.New
        self.features = []
        self.smooth_feat = None
        if feature is not None:
            self.features.append(feature)
            self.smooth_feat = feature

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * detection.feature
        self.hits += 1
        self.time_since_update = 0
        # if self.state == myTrackState.Tentative and self.hits >= self._n_init:
        self.state = myTrackState.Tracked

    def mark_lossed(self):
        """Mark this track as missed (no association at the current time step).
        """
        self.state = myTrackState.Lost

    # def is_tentative(self):
    #     """Returns True if this track is tentative (unconfirmed).
    #     """
    #     return self.state == myTrackState.Tentative
    #
    # def is_confirmed(self):
    #     """Returns True if this track is confirmed."""
    #     return self.state == myTrackState.Confirmed
    #
    # def is_deleted(self):
    #     """Returns True if this track is dead and should be deleted."""
    #     return self.state == myTrackState.Deleted




class myTracker:
    def __init__(self, max_iou_distance=0.7, max_age=30):
        g_gallery.set_max_age(max_age)
        self.max_iou_distance = max_iou_distance
        self.kf = kalman_filter.KalmanFilter()
        # self.tracks = []
        self.frame_id = 0
        self.tracked_tracks = []  # type: list[STrack]
        self.lost_tracks = []  # type: list[STrack]
        self.removed_tracks = []  # type: list[STrack]


    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    '''
    todo: update gallery
    use detections and tracks which have been predicted to update
    1. match detection by feature 
    2.1 cascade match , deepsort
    2.2 smooth feature, JDE IOU associate
    3 with gallery
    
    
    '''
    def update(self, detections, cost_limit=0.7, method='JDE'):#conf, bounding box

        #joint tracked and lost track
        exist = {}
        tracks = []
        for tk in self.tracked_tracks:
            exist[tk.track_id] = 1
            tracks.append(tk)
        for tk in self.lost_tracks:
            tid = tk.track_id
            if not exist.get(tid, 0):
                exist[tid] = 1
                tracks.append(tk)

        #get smooth feature of tracks above and detection features
        track_features = np.asarray([track.smooth_feat for track in tracks ])
        det_features = np.asarray([det.feature for det in detections])

        #如果当前没有正在追踪myTrack，那么直接通过detection来initiate一个track
        if len(track_features) == 0:
            #todo: initiate new track
            for i in range(len(detections)):
                self.initiate(detections[i])
            return
        match1, unmatched_track1, unmatched_detection1= [], [], []
        #step 1: associate by feature

        if len(track_features) > 0 and len(det_features) > 0:
            assert track_features.shape[1] == det_features.shape[1]
            #todo: choose the track and detection witch conf bigger than thresh
            #todo: 看是否有必要规范距离矩阵
            cost_matrix = np.dot(track_features, det_features.T) # distance matrix
            # cost_matrix = np.max(cost_matrix, 0)
            cost_matrix = 1 - cost_matrix
            #todo: fuse motion?

            #function: linear_assignment
            if cost_matrix.size == 0:
                return np.empty(shape=(0,2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

            cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=cost_limit)
            for ix, mx in enumerate(x):
                if mx >= 0:
                    match1.append([ix, mx])
            unmatched_track1 = np.where(x<0)[0]
            unmatched_detection1 = np.where(y<0)[0]
            match1 = np.array(match1)


        #step 2: associate by gallery


        #step 3: associate by iou distance

        for trackid, detid in match1:
            self.tracks[trackid].update(self.kf, detections[detid])

        for trackid in unmatched_track1:
            self.tracks[trackid].mark_lossed()

        for udet in unmatched_detection1:
            self.initiate(detections[udet])

        # return match1

    def initiate(self,detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(myTrack(mean, covariance, myTrack.next_id, n_init=3, max_age=30, feature=detection.feature))
        myTrack.next_id += 1