import time

import cv2
import seaborn as sns

from .object_detection.detection import YOLODetector
from .object_tracking.trackers import NaiveTracker, ObjectTracker
from .video_streaming import VideoStream
from typing import Dict

from .utils.image_utils import resize_with_aspect_ratio


def process_frame(
        frame, 
        detector: YOLODetector, 
        trackers: Dict[str, ObjectTracker], 
        class_to_color_and_name,
        ):
    bboxes_with_class = detector.predict(frame)
    bboxes = [bbox[:-1] for bbox in bboxes_with_class]
    processed_frames = []
    for tracker_name, tracker in trackers.items():
        frame_copy = frame.copy()
        tracker.update(bboxes)
        plot_bboxes(frame_copy, bboxes_with_class, class_to_color_and_name)
        plot_tracking(frame_copy, tracker, tracker_name)
        processed_frames.append(frame_copy)
    processed_frame = cv2.hconcat(processed_frames)
    return processed_frame


def fps_to_interval(fps: float):
    return 1.0 / fps


def wait_for_next_frame(desired_interval: float, start_time: float) -> float:
    current_time = time.time()
    interval = current_time - start_time
    time.sleep(max(0, desired_interval - interval))
    return time.time()


def plot_tracking(frame, tracker, tracker_name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    color = (0, 0, 255)
    cv2.putText(frame, tracker_name, (10, 30), font, font_scale, color, thickness)
    for object_id, (x, y) in tracker.objects.items():
        text = f"ID {object_id}"
        cv2.putText(
            frame,
            text,
            (x - 10, y - 10),
            font,
            font_scale,
            color,
            thickness,
        )
        cv2.circle(frame, (x, y), 4, color, -1)


def make_class_to_color_and_name(class_id_to_name):
    def hex_to_bgr(hex_color):
        hex_color = hex_color.lstrip("#")
        rgb_color = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
        return bgr_color

    colors_hex = sns.color_palette("hls", len(class_id_to_name)).as_hex()
    colors_bgr = [hex_to_bgr(color_hex) for color_hex in colors_hex]
    return {
        class_id: (colors_bgr[class_id], name)
        for class_id, name in class_id_to_name.items()
    }


def plot_bboxes(frame, bboxes, class_to_color_and_name):
    for x1, y1, x2, y2, class_ in bboxes:
        color, name = class_to_color_and_name[class_]
        cv2.putText(
            frame,
            name,
            (x1 - 10, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def process_video(video_path: str, use_kdtree: bool = False, desired_fps: float = 30.0):
    # Setup video stream
    video_stream = VideoStream.from_file(video_path)
    video_stream.start()

    # Setup Object Detector
    object_detector = YOLODetector()

    # Setup Object Trackers
    object_trackers = {
        "tracker 1": NaiveTracker(use_kdtree=True), 
        "tracker 2": NaiveTracker(use_kdtree=False),
    }

    # Setup class_to_color
    class_to_color_and_name = make_class_to_color_and_name(object_detector.class_id_to_name)

    desired_interval = fps_to_interval(desired_fps)
    start_time = time.time()
    while True:
        start_time = wait_for_next_frame(desired_interval, start_time)
        frame = video_stream.get_last_frame()
        frame = resize_with_aspect_ratio(frame, target_width=600)
        if frame is None:
            break
        processed_frame = process_frame(
            frame, 
            detector=object_detector, 
            trackers=object_trackers, 
            class_to_color_and_name=class_to_color_and_name,
        )
        cv2.imshow("Frame", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            video_stream.stop()
            break
    cv2.destroyAllWindows()
