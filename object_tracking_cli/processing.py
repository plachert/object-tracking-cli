import time
from typing import Dict

import cv2
import seaborn as sns

from .object_detection.detection import YOLODetector
from .object_tracking.base_tracker import ObjectTracker
from .object_tracking.trackers import AVAILABLE_TRACKERS
from .plotting import plot_bboxes, plot_tracking
from .utils.image_utils import resize_with_aspect_ratio
from .video_streaming import VideoStream


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


def process_video(video_path: str, config):
    # Setup video stream
    video_stream = VideoStream.from_file(video_path)
    video_stream.start()

    # Setup Object Detector
    object_detector = YOLODetector(**config["detection"])

    # Setup Object Trackers
    object_trackers = {}
    for tracker in config["trackers"]:
        tracker_name, params = next(iter(tracker.items()))
        name = f"{tracker_name} params: {params}"
        object_trackers[name] = AVAILABLE_TRACKERS[tracker_name](**params)

    # Setup class_to_color
    class_to_color_and_name = make_class_to_color_and_name(
        object_detector.class_id_to_name
    )

    desired_interval = fps_to_interval(config["video"]["desired_fps"])
    start_time = time.time()
    while True:
        start_time = wait_for_next_frame(desired_interval, start_time)
        frame = video_stream.get_last_frame()
        if frame is None:
            break
        frame = resize_with_aspect_ratio(
            frame, target_width=config["video"]["output_width"]
        )
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
