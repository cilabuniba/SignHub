from pytube import Playlist, YouTube
from downloader.logger import logger
from downloader.utils.utils import std_str


def generate_from_playlist(playlist: Playlist):
    annotations = []
    for url in playlist:
        youtube = YouTube(url)
        annotations.append(
            {
                "title": std_str(youtube.title),
                "url": url,
            }
        )
    logger.info(f"Generate {len(annotations)} videos annotations")
    return annotations


def generate_from_video(video: YouTube):
    annotations = [
        {
            "title": std_str(video.title),
            "url": video.watch_url,
        }
    ]
    logger.info(f"Generate video annotations")
    return annotations
