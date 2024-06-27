import subprocess
import os

from .parser.args import init_parser


def inference_hands(file_path):
    subprocess.run(["python", "pose-estimation/inference_hands.py", "hamer", file_path])


def inference_body(file_path):
    subprocess.run(["python", "pose-estimation/inference_body.py", "hybrik", file_path])


def recursive_folder(folder_path):
    return [
        os.path.join(folder_path, folder)
        for folder in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, folder))
    ]


def run_inference_from_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        print(f'Processing file: {file_path}')
        if os.path.isfile(file_path):
            if args.infence_type == "hands":
                inference_hands(file_path)
            elif args.infence_type == "body":
                inference_body(file_path)
            elif args.infence_type == "all":
                inference_body(file_path)
                inference_hands(file_path)


if __name__ == "__main__":
    args = init_parser()
    for folder in set(recursive_folder(args.folder)) - set(
        recursive_folder(args.folder_done)
    ):
        run_inference_from_folder(folder)
    print(f"Finito tutto!")
