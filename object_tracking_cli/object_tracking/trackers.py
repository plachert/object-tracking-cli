import logging
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as dist

from .base_tracker import ObjectTracker
from .utils.costs import iou_cost
from .utils.distance_utils import find_unique_closest_pairs

logger = logging.getLogger(__name__)
AVAILABLE_TRACKERS = {}


def register_tracker(cls):
    AVAILABLE_TRACKERS[cls.__name__] = cls
    return cls


@register_tracker
class MotionAgnosticTracker(ObjectTracker):
    def __init__(
        self,
        assignment_strategy: Literal["naive", "kd_tree", "hangarian"] = "naive",
        cost: Literal["euclidean", "iou"] = "euclidean",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.assignment_strategy = assignment_strategy
        self.cost = cost
        if assignment_strategy == "kd_tree" and cost == "iou":
            raise ValueError("Only euclidean cost works with KD-tree")

    def _post_assignment(self, matches, bboxes, object_ids, registered_bboxes):
        used_registered_bbox_idx = set()
        used_bboxes_idx = set()
        for bbox_idx, registered_bbox_idx in matches.items():
            self._objects[object_ids[registered_bbox_idx]] = bboxes[bbox_idx]
            self._missing_frames[object_ids[registered_bbox_idx]] = 0
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

    def _make_cost_matrix(self, bboxes, registered_bboxes):
        if self.cost == "euclidean":
            registered_centroids = self._bboxes_to_centroids(registered_bboxes)
            bbox_centroids = self._bboxes_to_centroids(bboxes)
            cost_matrix = dist.cdist(np.array(registered_centroids), bbox_centroids)
        elif self.cost == "iou":
            cost_matrix = np.zeros((len(registered_bboxes), len(bboxes)))
            for i, bbox1 in enumerate(registered_bboxes):
                for j, bbox2 in enumerate(bboxes):
                    cost_matrix[i, j] = iou_cost(bbox1, bbox2)
        return cost_matrix

    def _naive_assignement(self, bboxes, object_ids, registered_bboxes):
        # https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/ approach
        cost_matrix = self._make_cost_matrix(bboxes, registered_bboxes)
        rows = cost_matrix.min(axis=1).argsort()
        cols = cost_matrix.argmin(axis=1)[rows]
        usedRows = set()
        usedCols = set()
        matches = {}
        for row, col in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            matches[col] = row
            usedRows.add(row)
            usedCols.add(col)
        self._post_assignment(matches, bboxes, object_ids, registered_bboxes)

    def _kd_tree_assignment(self, bboxes, object_ids, registered_bboxes):
        registered_centroids = self._bboxes_to_centroids(registered_bboxes)
        bbox_centroids = self._bboxes_to_centroids(bboxes)
        # my approach based on KD-tree
        matches = find_unique_closest_pairs(
            registered_centroids, bbox_centroids
        )  # new point idx to centroid idx
        self._post_assignment(matches, bboxes, object_ids, registered_bboxes)

    def _hungarian_assignment(self, bboxes, object_ids, registered_bboxes):
        def make_matrix_square(matrix, pad_value=0):
            rows, cols = matrix.shape
            max_dim = max(rows, cols)
            pad_rows = max_dim - rows
            pad_cols = max_dim - cols
            if pad_rows > 0:
                padding = np.full((pad_rows, cols), pad_value)
                matrix = np.vstack((matrix, padding))

            if pad_cols > 0:
                padding = np.full((rows, pad_cols), pad_value)
                matrix = np.hstack((matrix, padding))
            return matrix

        cost_matrix = self._make_cost_matrix(bboxes, registered_bboxes)
        cost_matrix_square = make_matrix_square(cost_matrix)
        row_indices, col_indices = linear_sum_assignment(cost_matrix_square)
        org_rows, org_cols = cost_matrix.shape
        matches = {}
        for row, col in zip(row_indices, col_indices):
            if row < org_rows and col < org_cols:
                matches[col] = row
        self._post_assignment(matches, bboxes, object_ids, registered_bboxes)

    def _handle_assignments(self, bboxes, object_ids, registered_bboxes):
        if self.assignment_strategy == "kd_tree":
            self._kd_tree_assignment(bboxes, object_ids, registered_bboxes)

        elif self.assignment_strategy == "naive":
            self._naive_assignement(bboxes, object_ids, registered_bboxes)

        elif self.assignment_strategy == "hungarian":
            self._hungarian_assignment(bboxes, object_ids, registered_bboxes)
