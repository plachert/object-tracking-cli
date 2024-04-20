import logging

import click

from .processing import process_video

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@click.command()
@click.argument(
    "video_file",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    required=True,
)
@click.option(
    "--use-kdtree",
    is_flag=True,
    help="Use KD-tree for tracking.",
)
def cli(video_file, use_kdtree):
    process_video(video_file, use_kdtree)
