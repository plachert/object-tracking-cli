from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List

from ..object_detection.detection import Bbox_xyxy


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
