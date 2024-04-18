from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List

import numpy as np
from scipy.spatial import distance as dist

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

    @abstractmethod
    def update(self, bboxes_xyxy: List[Bbox_xyxy]):  # noqa
        """Assign, register or deregister objects based on last detections."""


class NaiveTracker(ObjectTracker):
    def update(self, rects):
        if len(rects) == 0:
            to_deregister = []
            for objectID in self._missing_frames.keys():
                self._missing_frames[objectID] += 1
                if self._missing_frames[objectID] > self._max_missing_frames:
                    to_deregister.append(objectID)
            for id_ in to_deregister:
                self.deregister_object(id_)
            return self._objects
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self._objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register_object(inputCentroids[i])

        else:
            objectIDs = list(self._objects.keys())
            objectCentroids = list(self._objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self._objects[objectID] = inputCentroids[col]
                self._missing_frames[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self._missing_frames[objectID] += 1

                    if self._missing_frames[objectID] > self._max_missing_frames:
                        self.deregister_object(objectID)

            else:
                for col in unusedCols:
                    self.register_object(inputCentroids[col])
        return self._objects


if __name__ == "__main__":
    tracker = NaiveTracker()
    tracker.update([(1, 1, 2, 2), (5, 1, 6, 2)])
    tracker.update([(4, 1, 5, 2), (6, 1, 7, 2)])
    print(tracker.objects)
