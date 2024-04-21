from typing import List

import numpy as np

from ...object_detection.detection import Bbox_xyxy_with_class_and_score


def get_centroids(bboxes: List[Bbox_xyxy_with_class_and_score]) -> np.ndarray:
    bboxes_array = np.array(bboxes)
    x1 = bboxes_array[:, 0]
    y1 = bboxes_array[:, 1]
    x2 = bboxes_array[:, 2]
    y2 = bboxes_array[:, 3]
    centroid_x = (x1 + x2) / 2
    centroid_y = (y1 + y2) / 2
    centroids = np.column_stack((centroid_x, centroid_y))
    return centroids


def euclidean_cost(
    bbox_1: Bbox_xyxy_with_class_and_score,
    bbox_2: Bbox_xyxy_with_class_and_score,
) -> float:
    centroid_1 = get_centroid(bbox_1)
    centroid_2 = get_centroid(bbox_2)
    cost = np.linalg.norm(centroid_1 - centroid_2)
    return cost


def iou_cost(
    bbox_1: Bbox_xyxy_with_class_and_score,
    bbox_2: Bbox_xyxy_with_class_and_score,
) -> float:
    x1, y1, x2, y2, _, _ = bbox_1
    x3, y3, x4, y4, _, _ = bbox_2
    x_intersection = max(x1, x3)
    y_intersection = max(y1, y3)
    x_intersection_right = min(x2, x4)
    y_intersection_bottom = min(y2, y4)
    intersection_area = max(0, x_intersection_right - x_intersection) * max(
        0, y_intersection_bottom - y_intersection
    )
    area_bbox1 = (x2 - x1) * (y2 - y1)
    area_bbox2 = (x4 - x3) * (y4 - y3)
    union_area = area_bbox1 + area_bbox2 - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou
