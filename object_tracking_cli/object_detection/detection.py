import pathlib
from typing import List, Tuple

from ultralytics import YOLO

Bbox_xyxy = Tuple[int, int, int, int]  # top-left, bottom-right
Bbox_xyxy_with_class = Tuple[int, int, int, int, int]
Bbox_xyxy_with_class_and_score = Tuple[int, int, int, int, int, float]

HERE = pathlib.Path(__file__).parent


class YOLODetector:
    def __init__(self, conf=0.3, iou=0.7) -> None:
        self.model = YOLO(HERE / "yolov3-tinyu.pt")
        self.model.TASK = "detect"
        self.class_id_to_name = self.model.model.names
        self.predict_cfg = {"conf": conf, "iou": iou, "verbose": False}

    @property
    def available_classes(self):
        return self.class_id_to_name.values()

    def predict(self, frame) -> List[Bbox_xyxy_with_class]:
        results = self.model.predict(frame, **self.predict_cfg)[0]
        bboxes_xyxy_with_class_and_score = (
            self._yolo_bboxes_to_bboxes_xyxy_with_classes_and_scores(results.boxes)
        )
        return bboxes_xyxy_with_class_and_score

    def get_class_name(self, bbox: Bbox_xyxy_with_class):
        return self.class_id_to_name[bbox[-1]]

    def _yolo_bboxes_to_bboxes_xyxy_with_classes_and_scores(self, yolo_bboxes):
        def to_tuple(bbox):
            xyxy = bbox.xyxy.numpy()[0]
            x1, y1, x2, y2 = xyxy
            class_ = bbox.cls.numpy()[0]
            score = bbox.conf.numpy()[0]
            return (int(x1), int(y1), int(x2), int(y2), int(class_), score)

        bboxes_xyxy_with_classes = [to_tuple(bbox) for bbox in yolo_bboxes]
        return bboxes_xyxy_with_classes
