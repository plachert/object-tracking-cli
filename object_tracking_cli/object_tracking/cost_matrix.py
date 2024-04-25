from typing import Callable, List

import numpy as np
from scipy.spatial import distance as dist

from ..object_detection.detection import Bbox_xyxy_with_class_and_score
from .utils.bbox import calc_centroids, iou

CostMatrixFunction = Callable[
    [List[Bbox_xyxy_with_class_and_score], List[Bbox_xyxy_with_class_and_score]],
    np.ndarray,
]

AVAILABLE_COST_MATRIX_FUNCS = {}


def register_func(cls):
    AVAILABLE_COST_MATRIX_FUNCS[cls.__name__] = cls
    return cls


@register_func
def euclidean_cost_matrix(
    bboxes: List[Bbox_xyxy_with_class_and_score],
    registered_bboxes: List[Bbox_xyxy_with_class_and_score],
):
    registered_centroids = calc_centroids(registered_bboxes)
    bbox_centroids = calc_centroids(bboxes)
    cost_matrix = dist.cdist(np.array(registered_centroids), bbox_centroids)
    return cost_matrix


@register_func
def iou_cost_matrix(
    bboxes: List[Bbox_xyxy_with_class_and_score],
    registered_bboxes: List[Bbox_xyxy_with_class_and_score],
):
    cost_matrix = np.zeros((len(registered_bboxes), len(bboxes)))
    for i, bbox1 in enumerate(registered_bboxes):
        for j, bbox2 in enumerate(bboxes):
            print(bbox1, bbox2)
            cost_matrix[i, j] = 1.0 - iou(bbox1, bbox2)
    return cost_matrix
