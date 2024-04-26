from collections import OrderedDict
from functools import partial
from typing import Dict, List

from ..object_detection.detection import Bbox_xyxy_with_class_and_score
from .assignment import (
    AVAILABLE_ASSIGNMENT_FUNCS,
    AssignmentFunction,
    hungarian_assignment,
)
from .cost_matrix import (
    AVAILABLE_COST_MATRIX_FUNCS,
    CostMatrixFunction,
    euclidean_cost_matrix,
)
from .motion_model import AVAILABLE_MOTION_MODELS, MotionAgnosticModel, MotionModel
from .utils.bbox import calc_centroids


class MultiObjectTracker:
    def __init__(
        self,
        assignment_func: AssignmentFunction = hungarian_assignment,
        cost_matrix_func: CostMatrixFunction = euclidean_cost_matrix,
        motion_model_cls: MotionModel = MotionAgnosticModel,
        max_missing_frames: int = 3,
    ):
        self._max_missing_frames = max_missing_frames
        self.assignment_func = assignment_func
        self.cost_matrix_func = cost_matrix_func
        self.motion_model_cls = motion_model_cls
        self._objects = OrderedDict()
        self._missing_frames = OrderedDict()
        self._next_object_id = 0

    @classmethod
    def from_config(cls, config: Dict):
        assignment_type, assignment_params = next(
            iter(config["assignment_func"].items())
        )
        assignment_params = {} if assignment_params is None else assignment_params
        cost_type, cost_params = next(iter(config["cost_matrix_func"].items()))
        cost_params = {} if cost_params is None else cost_params
        model_type, model_params = next(iter(config["motion_model_cls"].items()))
        model_params = {} if model_params is None else model_params
        assignment_func = partial(
            AVAILABLE_ASSIGNMENT_FUNCS[assignment_type], **assignment_params
        )
        cost_matrix_func = partial(
            AVAILABLE_COST_MATRIX_FUNCS[cost_type], **cost_params
        )
        motion_model_cls = partial(AVAILABLE_MOTION_MODELS[model_type], **model_params)

        return cls(
            assignment_func=assignment_func,
            cost_matrix_func=cost_matrix_func,
            motion_model_cls=motion_model_cls,
            max_missing_frames=config["max_missing_frames"],
        )

    @property
    def object_centroids(self):
        object_centroids = {}
        ids = self.objects.keys()
        centroids = calc_centroids(self.objects.values())
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
            for object_id in self._missing_frames.keys():
                self._missing_frames[object_id] += 1
                if self._missing_frames[object_id] > self._max_missing_frames:
                    to_deregister.append(object_id)
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

    def _post_assignment(self, assignments, bboxes, object_ids, registered_bboxes):
        used_registered_bbox_idx = set()
        used_bboxes_idx = set()
        for bbox_idx, registered_bbox_idx in assignments.items():
            object_id = object_ids[registered_bbox_idx]
            self._objects[object_id] = bboxes[bbox_idx]
            self._missing_frames[object_id] = 0
            used_registered_bbox_idx.add(registered_bbox_idx)
            used_bboxes_idx.add(bbox_idx)
        unused_registered_bboxes_idx = (
            set(range(len(registered_bboxes))) - used_registered_bbox_idx
        )
        unused_bboxes_idx = set(range(len(bboxes))) - used_bboxes_idx
        for unused_registered_bbox_idx in unused_registered_bboxes_idx:
            self.handle_missing(object_ids[unused_registered_bbox_idx])
        for unused_bbox_idx in unused_bboxes_idx:
            self.register_object(bboxes[unused_bbox_idx])

    def _handle_assignments(
        self,
        bboxes: List[Bbox_xyxy_with_class_and_score],
        object_ids: List[int],
        registered_bboxes: List[Bbox_xyxy_with_class_and_score],
    ):  # noqa
        cost_matrix = self.cost_matrix_func(bboxes, registered_bboxes)
        assignments = self.assignment_func(cost_matrix)
        self._post_assignment(assignments, bboxes, object_ids, registered_bboxes)
