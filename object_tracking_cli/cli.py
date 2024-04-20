import logging
import pathlib

import click
import yaml

from .processing import process_video

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

HERE = pathlib.Path(__file__).parent
DEFAULT_CONFIG_FILE = HERE / "configs/default_config.yaml"


def load_config(path=DEFAULT_CONFIG_FILE):
    with open(path, "r") as f:
        return yaml.safe_load(f)


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
    "--config",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to YAML configuration file.",
)
def cli(video_file, config):
    if config:
        config = load_config(config)
    else:
        config = load_config()
    process_video(video_file, config)
