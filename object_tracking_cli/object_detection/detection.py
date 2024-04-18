import pathlib
from typing import List, Tuple

from ultralytics import YOLO

Bbox_xyxy = Tuple[int, int, int, int]  # top-left, bottom-right
Bbox_xyxy_with_class = Tuple[Bbox_xyxy, int]

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
        bboxes_xyxy_with_class = self._yolo_bboxes_to_bboxes_xyxy_with_classes(
            results.boxes
        )
        return bboxes_xyxy_with_class

    def get_class_name(self, bbox: Bbox_xyxy_with_class):
        return self.class_id_to_name[bbox[-1]]

    def _yolo_bboxes_to_bboxes_xyxy_with_classes(self, yolo_bboxes):
        def to_tuple(bbox):
            return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

        bboxes_xyxy_with_classes = [
            (*to_tuple(bbox.xyxy.numpy()[0]), int(bbox.cls.numpy()[0]))
            for bbox in yolo_bboxes
        ]
        return bboxes_xyxy_with_classes
