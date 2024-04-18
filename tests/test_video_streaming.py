import pathlib
import time

import pytest

from object_tracking_cli.video_streaming import UnsupportedVideoFormat, VideoStream
from tests.setup import DURATION, FPS, VALID_VIDEO, make_test_video


@pytest.fixture(scope="function")
def video_stream():
    if not pathlib.Path(VALID_VIDEO).exists():
        make_test_video()
    return VideoStream.from_file(VALID_VIDEO, buffer_size=10)


def test_from_file_success(video_stream):
    assert isinstance(video_stream, VideoStream)
    assert video_stream.stream.isOpened() == True


def test_from_file_nonexistent(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoStream.from_file(str(tmp_path / "nonexistent.mp4"))


def test_from_file_unsupported_format(tmp_path):
    file_path = str(tmp_path / "invalid_ext.abc")
    with open(file_path, "w"):
        pass
    with pytest.raises(UnsupportedVideoFormat):
        VideoStream.from_file(file_path)


def _get_number_of_processed_frames(video_stream, processing_time: float = 0.0) -> int:
    video_stream.start()
    grabbed_frames = 0
    while True:
        frame = video_stream.get_last_frame()
        time.sleep(processing_time)
        if frame is None:
            break
        grabbed_frames += 1
    return grabbed_frames


def test_no_of_frames_faster_processing(video_stream):
    expected = DURATION * FPS
    grabbed_frames = _get_number_of_processed_frames(video_stream, 0.0)
    assert grabbed_frames == expected


def test_no_of_frames_faster_grabbing(video_stream):
    expected = DURATION * FPS
    grabbed_frames = _get_number_of_processed_frames(video_stream, 0.2)
    assert grabbed_frames == expected


def test_interruption(video_stream):
    video_stream.start()
    video_stream.stop()
    assert video_stream.stopped
    assert video_stream.thread.is_alive() == False
