from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List

import numpy as np

from ..object_detection.detection import Bbox_xyxy_with_class_and_score


class ObjectTracker(ABC):
    def __init__(self, max_missing_frames: int = 3):
        self._max_missing_frames = max_missing_frames
        self._objects = OrderedDict()
        self._missing_frames = OrderedDict()
        self._next_object_id = 0

    @property
    def object_centroids(self):
        object_centroids = {}
        ids = self.objects.keys()
        centroids = self._bboxes_to_centroids(self.objects.values())
        for id_, centroid in zip(ids, centroids):
            object_centroids[id_] = centroid
        return object_centroids

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

    def update(self, bboxes: List[Bbox_xyxy_with_class_and_score]):
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

        # no registered objects. Register all new bboxes
        if len(self._objects) == 0:
            for bbox in bboxes:
                self.register_object(bbox)

        else:
            object_ids = list(self._objects.keys())
            registerd_bboxes = list(self._objects.values())
            self._handle_assignments(bboxes, object_ids, registerd_bboxes)
        return self._objects

    def _bboxes_to_centroids(self, bboxes: List[Bbox_xyxy_with_class_and_score]):
        centroids = np.zeros((len(bboxes), 2), dtype="int")
        for i, (x1, y1, x2, y2, _, _) in enumerate(bboxes):
            center_x = int((x1 + x2) / 2.0)
            center_y = int((y1 + y2) / 2.0)
            centroids[i] = (center_x, center_y)
        return centroids

    @abstractmethod
    def _handle_assignments(
        self,
        bboxes: List[Bbox_xyxy_with_class_and_score],
        object_ids: List[int],
        registered_bboxes: List[Bbox_xyxy_with_class_and_score],
    ):  # noqa
        """Assign, register or deregister objects based on last detections."""
