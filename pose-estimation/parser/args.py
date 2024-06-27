import argparse


def init_parser():
    parser = argparse.ArgumentParser(description="Estimate pose of a person in a video")
    parser.add_argument(
        "folder",
        type=str,
        help="folder containing the video files",
    )
    parser.add_argument(
        "folder_done",
        type=str,
        help="folder to save the output files",
    )
    parser.add_argument(
        "--infence_type",
        type=str,
        default="all",
        choices=["hands", "body", "all"],
        help="type of inference to run",
    )
    return parser.parse_args()
