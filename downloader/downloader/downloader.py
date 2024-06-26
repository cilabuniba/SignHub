from logger import logger
from downloader.utils.utils import std_str
from concurrent.futures import ThreadPoolExecutor
import yt_dlp
import os


class DownloaderAudio:
    def __init__(self, output_path):
        self.output_path = output_path

    def download_audios(self, urls, max_workers=5):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._download_audio, urls))

    def download_yt_audio(self, urls):
        try:
            self._download_audio(urls, self.output_path)
        except:
            logger.error(f"Error downloading: {urls}")
        _rename_files(self.output_path)

    def _download_audio(self, urls):
        ydl_opts = {
            "format": "bestaudio",
            "outtmpl": self.output_path + "%(title)s.%(ext)s",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                },
            ],
            "quiet": True,
        }
        _download_url(urls, ydl_opts)


class DownloaderVideo:
    def __init__(self, output_path):
        self.output_path = output_path

    def download_videos(self, urls, max_workers=5):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._download_video, urls))

    def download_yt_video(self, url):
        try:
            self._download_video(url, self.output_path)
        except:
            logger.error(f"Error downloading: {url}")
        _rename_files(self.output_path)

    def _download_video(self, url):
        ydl_opts = {
            "format": "bestvideo",
            "outtmpl": self.output_path + "%(title)s.%(ext)s",
            "postprocessors": [
                {
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                }
            ],
            "quiet": True,
        }
        _download_url(url, ydl_opts)


def _rename_files(output_path):
    files = os.listdir(output_path)
    for file in files:
        old_file_path = os.path.join(output_path, file)
        new_file_path = os.path.join(output_path, std_str(file))
        os.rename(old_file_path, new_file_path)


def _download_url(url, ydl_opts):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(url)
