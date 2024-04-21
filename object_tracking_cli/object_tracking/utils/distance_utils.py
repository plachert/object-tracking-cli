from typing import List

import numpy as np
from scipy.spatial import KDTree

from ...object_detection.detection import Bbox_xyxy_with_class_and_score


def find_unique_closest_pairs(centroids, points):
    centroid_tree = KDTree(centroids)
    closest_centroids_indices = centroid_tree.query(points)[1]
    used_centroid_idxs = set()
    closest_pairs = {}
    for point_idx, centroid_idx in enumerate(closest_centroids_indices):
        # resolve potential conflict
        if centroid_idx in used_centroid_idxs:
            k = 2
            while k < len(centroids):
                next_centroid_idx = centroid_tree.query(points[point_idx], k=k)[1][
                    k - 1
                ]
                if next_centroid_idx not in used_centroid_idxs:
                    closest_pairs[point_idx] = next_centroid_idx
                    used_centroid_idxs.add(next_centroid_idx)
                    break
                k += 1
        else:
            closest_pairs[point_idx] = centroid_idx
            used_centroid_idxs.add(centroid_idx)
    return closest_pairs


def create_cost_matrix(
    bboxes_1: List[Bbox_xyxy_with_class_and_score],
    bboxes_2: List[Bbox_xyxy_with_class_and_score],
    cost_function,
) -> np.ndarray:
    bboxes_1 = np.array(bboxes_1)
    bboxes_2 = np.array(bboxes_2)
    distance_matrix = cost_function(bboxes_1[:, None], bboxes_2)
    return distance_matrix
