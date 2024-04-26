import numpy as np
from filterpy.kalman import KalmanFilter

from ..object_detection.detection import Bbox_xyxy_with_class_and_score
from .utils.bbox import calc_centroids


class MotionAgnosticModel:
    def __init__(self, bbox: Bbox_xyxy_with_class_and_score) -> None:
        self.bbox = bbox

    def predict_bbox(self):
        return self.bbox

    def update_bbox(self, measurement: Bbox_xyxy_with_class_and_score):
        return measurement


class KFCentroidVelocityModel:
    def __init__(self, bbox: Bbox_xyxy_with_class_and_score) -> None:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        centroid = calc_centroids([bbox])[0]
        kf.x = np.array((centroid[0], centroid[1], 0, 0))
        kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        kf.P *= 1000
        kf.R *= 10
        self.kf = kf
        self.bbox = bbox
        self.centroid = centroid

    def predict_bbox(self):
        """Update state based on the velocity model and return the updated bbox."""
        self.kf.predict()
        centroid = self.kf.x[:2]
        self._reposition_bbox_and_centroid(centroid)
        return self.bbox

    def update_bbox(self, measurement: Bbox_xyxy_with_class_and_score):
        """Refine the estimate based on the measurement and return the refined bbox."""
        centroid = calc_centroids([measurement])[0]
        self.kf.update(centroid)
        centroid = self.kf.x[:2]
        self._reposition_bbox_and_centroid(centroid)
        return self.bbox

    def _reposition_bbox_and_centroid(self, new_centroid):
        x1, y1, x2, y2, score, class_ = self.bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        x1 = new_centroid[0] - (bbox_width // 2)
        y1 = new_centroid[1] - (bbox_height // 2)
        x2 = new_centroid[0] + (bbox_width // 2)
        y2 = new_centroid[1] + (bbox_height // 2)
        self.bbox = (x1, y1, x2, y2, score, class_)
        self.centroid = new_centroid
