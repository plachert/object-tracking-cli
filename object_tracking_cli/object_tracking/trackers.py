from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List

import numpy as np
from scipy.spatial import distance as dist

from ..object_detection.detection import Bbox_xyxy
from .distance_utils import find_unique_closest_pairs


class ObjectTracker(ABC):
    def __init__(self, max_missing_frames: int = 3):
        self._max_missing_frames = max_missing_frames
        self._objects = OrderedDict()
        self._missing_frames = OrderedDict()
        self._next_object_id = 0

    @property
    def objects(self):
        return self._objects

    @property
    def no_objects(self):
        return len(self._objects)

    def register_object(self, object_):
        self._objects[self._next_object_id] = object_
        self._missing_frames[self._next_object_id] = 0
        self._next_object_id += 1

    def deregister_object(self, object_id: int):
        del self._objects[object_id]
        del self._missing_frames[object_id]

    def handle_missing(self, object_id: int):
        self._missing_frames[object_id] += 1
        if self._missing_frames[object_id] >= self._max_missing_frames:
            self.deregister_object(object_id)

    @abstractmethod
    def update(self, bboxes_xyxy: List[Bbox_xyxy]):  # noqa
        """Assign, register or deregister objects based on last detections."""


class NaiveTracker(ObjectTracker):
    def __init__(self, use_kdtree: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_kdtree = use_kdtree

    def _bboxes_to_centroids(self, bboxes: List[Bbox_xyxy]):
        centroids = np.zeros((len(bboxes), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            center_x = int((x1 + x2) / 2.0)
            center_y = int((y1 + y2) / 2.0)
            centroids[i] = (center_x, center_y)
        return centroids

    def _handle_assignments(self, bbox_centroids, object_ids, centroids):
        if self.use_kdtree:
            # my approach based on KD-tree
            matches = find_unique_closest_pairs(
                centroids, bbox_centroids
            )  # new point idx to centroid idx
            used_centroids_idx = set()
            used_bboxes_centroids_idx = set()
            for bbox_centroid_idx, centroid_idx in matches.items():
                self._objects[object_ids[centroid_idx]] = bbox_centroids[
                    bbox_centroid_idx
                ]
                self._missing_frames[object_ids[centroid_idx]] = 0
                used_centroids_idx.add(centroid_idx)
                used_bboxes_centroids_idx.add(bbox_centroid_idx)
            unused_centroids_idx = set(range(len(centroids))) - used_centroids_idx
            unused_bboxes_centroids_idx = (
                set(range(len(bbox_centroids))) - used_bboxes_centroids_idx
            )
            for unused_centroid_idx in unused_centroids_idx:
                self.handle_missing(object_ids[unused_centroid_idx])
            for unused_bbox_centroid_idx in unused_bboxes_centroids_idx:
                self.register_object(bbox_centroids[unused_bbox_centroid_idx])
        else:
            # https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/ approach
            D = dist.cdist(np.array(centroids), bbox_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = object_ids[row]
                self._objects[objectID] = bbox_centroids[col]
                self._missing_frames[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = object_ids[row]
                    self._missing_frames[objectID] += 1

                    if self._missing_frames[objectID] > self._max_missing_frames:
                        self.deregister_object(objectID)
            else:
                for col in unusedCols:
                    self.register_object(bbox_centroids[col])

    def update(self, bboxes):
        # no new bounding boxes
        if len(bboxes) == 0:
            to_deregister = []
            for objectID in self._missing_frames.keys():
                self._missing_frames[objectID] += 1
                if self._missing_frames[objectID] > self._max_missing_frames:
                    to_deregister.append(objectID)
            for id_ in to_deregister:
                self.deregister_object(id_)
            return self._objects

        # get centriods from bboxes
        bbox_centroids = self._bboxes_to_centroids(bboxes)

        # no registered objects. Register all new bboxes
        if len(self._objects) == 0:
            for i in range(0, len(bbox_centroids)):
                self.register_object(bbox_centroids[i])

        else:
            object_ids = list(self._objects.keys())
            centroids = list(self._objects.values())
            self._handle_assignments(bbox_centroids, object_ids, centroids)
        return self._objects
