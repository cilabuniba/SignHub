import argparse


def init_parser():
    parser = argparse.ArgumentParser(description="Downloader for yt videos")
    parser.add_argument(
        "--url",
        type=str,
        help="url of the video or playlist",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="playlist",
        help="type of the url",
    )
    parser.add_argument(
        "--download",
        type=bool,
        default=False,
        help="download the videos",
    )
    parser.add_argument(
        "--download_video",
        type=bool,
        help="download the videos",
    )
    parser.add_argument(
        "--download_audio",
        type=bool,
        help="download the audio",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/",
        help="output path",
    )
    parser.add_argument(
        "--file_json",
        type=str,
        default="annotations.json",
        required=False,
        help="path to the json file",
    )
    parser.add_argument(
        "--transcribe",
        type=bool,
        default=False,
        help="transcribe the audio inside the video",
    )
    parser.add_argument(
        "--model_dimension",
        type=str,
        default="base",
        help="model dimension",
    )

    return parser.parse_args()
