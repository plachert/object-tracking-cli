import logging

import click
from processing import process_video

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@click.command()
@click.argument(
    "video_file",
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    required=True,
)
def cli(video_file):
    process_video(video_file)


if __name__ == "__main__":
    cli()
