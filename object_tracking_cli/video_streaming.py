import logging
import pathlib
import time
from queue import Queue
from threading import Thread

import cv2

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".mp4", ".webm"]


class UnsupportedVideoFormat(Exception):
    pass


class VideoStream:
    def __init__(self, video_capture: cv2.VideoCapture, buffer_size: int = 128):
        self.stream = video_capture
        self.buffer = Queue(maxsize=buffer_size)
        self.stopped = False

    def start(self):
        logger.info("Starting VideoStream")
        self.thread = Thread(target=self.grab_frames, daemon=True)
        self.thread.start()

    def stop(self):
        logger.info("Stopping gracefully")
        self.stopped = True
        self.thread.join()

    def grab_frames(self):
        while not self.stopped:
            if self.buffer.full():
                time.sleep(0.1)
            else:
                is_grabbed, frame = self.stream.read()
                if is_grabbed:
                    self.buffer.put(frame)
                else:
                    self.stopped = True
        self.stream.release()

    def get_last_frame(self):
        while self.buffer.qsize() == 0:
            if self.stopped:
                self.stop()
                return
            time.sleep(0.0001)
        return self.buffer.get()

    @classmethod
    def from_file(cls, file_path: str, buffer_size: int = 128) -> cv2.VideoCapture:
        path = pathlib.Path(file_path)
        if not path.exists():
            raise FileNotFoundError("Video could not be found.")
        if path.suffix not in SUPPORTED_EXTENSIONS:
            raise UnsupportedVideoFormat(
                f"Only these formats are supported: {SUPPORTED_EXTENSIONS}"
            )
        cap = cv2.VideoCapture(file_path)
        assert cap.isOpened(), "The video could not be accessed for some reason"
        return cls(cap, buffer_size)
