import logging

from .processing import process_video
from .video_streaming import UnsupportedVideoFormat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def cli():
    while True:
        video_path = input("Enter the path to the video file: ")
        use_kdtree = (
            True
            if input("Use KD-Tree? Enter True or False: ").lower() == "true"
            else False
        )
        try:
            process_video(
                video_path,
                use_kdtree,
            )  # wide error scope. Refactor. TODO: process_video(video_cap)
        except FileNotFoundError as e:
            logger.error(f"{e} Try again")
        except UnsupportedVideoFormat as e:
            logger.error(f"{e} Try again")
        except AssertionError as e:
            logger.error(f"{e} Try again")


if __name__ == "__main__":
    cli()
