import argparse


def init_parser():
    parser = argparse.ArgumentParser(description="Estimate pose of a person in a video")
    parser.add_argument(
        "folder",
        type=str,
        help="folder containing the video files",
    )
    return parser.parse_args()
