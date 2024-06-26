from logger import logger
import os


def _make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"{directory} created")
    else:
        logger.info(f"{directory} folder already exists")


def make_download_directory(directory):
    _make_directory(directory)
    _make_directory(os.path.join(directory, "audio"))
    _make_directory(os.path.join(directory, "video"))
    _make_directory(os.path.join(directory, "video_cropped"))
    _make_directory(os.path.join(directory, "transcripts"))
    _make_directory(os.path.join(directory, "xlsx"))


def compare_and_merge_lists(old, new):
    if old != new:
        return old + new
    else:
        return new


def compare_list_to_folder_audio(list: list, folder: str):
    files_in_folder = [
        f[:-4] for f in os.listdir(folder) if f.endswith(".mp3")
    ]  # rimuove l'estensione .mp3

    missings = [item for item in list if item["title"] not in files_in_folder]
    return [item["url"] for item in missings]


def compare_list_to_folder_video(list: list, folder: str):
    files_in_folder = [
        f[:-4] for f in os.listdir(folder) if f.endswith(".mp4")
    ]  # rimuove l'estensione .mp4

    missings = [item for item in list if item["title"] not in files_in_folder]
    return [item["url"] for item in missings]


def std_str(string: str):
    return (
        string.replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("⧸", "_")
        .replace("｜", "")
        .replace("|", "")
        .replace("__", "_")
        .replace("__", "_")
    )
