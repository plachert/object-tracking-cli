from ...object_detection.detection import Bbox_xyxy_with_class_and_score


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
    iou_cost = -iou
    return iou_cost
