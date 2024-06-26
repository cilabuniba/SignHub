import subprocess
import os

from downloader.parser.args import init_parser


def inference_hands(file_path):
    subprocess.run(["python", "inference_hands.py", "hamer", file_path])


def inference_body(file_path):
    subprocess.run(["python", "inference_body.py", "hybrik", file_path])


def run_inference_from_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            inference_hands(file_path)
            inference_body(file_path)


if __name__ == "__main__":
    args = init_parser()
    for folder in args.folder:
        run_inference_from_folder(os.path.join(args.folder, folder))
    print(f"Finito tutto!")
