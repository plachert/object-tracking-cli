import numpy as np
from filterpy.kalman import KalmanFilter

from .utils.bbox import calc_centroids


class KFCentroidVelocityModel:
    def __init__(self) -> None:
        self.filters = {}

    def register_filter(self, object_, object_id):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        centroid = calc_centroids([object_])[0]
        kf.x = np.array((centroid[0], centroid[1], 0, 0))
        kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        kf.P *= 1000
        kf.R *= 10
        self.filters[object_id] = kf

    def deregister_filter(self, object_id):
        del self.filters[object_id]

    def update(self):
        for _, kf in self.filters.items():
            kf.predict()
