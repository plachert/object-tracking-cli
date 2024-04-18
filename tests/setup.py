import pathlib

import cv2
import numpy as np

VALID_VIDEO = str(pathlib.Path(__file__).parent / "test_video.mp4")
WIDTH = 640
HEIGHT = 480
FPS = 30
DURATION = 2


def make_test_video():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VALID_VIDEO, fourcc, FPS, (WIDTH, HEIGHT))
    for t in range(FPS * DURATION):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        x = int((WIDTH - 100) * t / (FPS * DURATION - 1))
        cv2.rectangle(
            frame, (x, WIDTH // 3), (x + 100, HEIGHT * 2 // 3), (255, 255, 255), -1
        )
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()
