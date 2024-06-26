import argparse
import os.path
import shutil

import yaml

from estimators.face.face_mediapipe_estimator import FaceMediapipeEstimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        default=None,
        type=str,
        help="model to use for hands pose estimation",
    )
    parser.add_argument(
        "vid",
        default=None,
        type=str,
        help="vid of the video (save on aws S3)",
    )
    parser.add_argument(
        "--config_file",
        default="configs/config.yml",
        type=str,
        help="path of the config file",
    )
    parser.add_argument(
        "--python_environment",
        default="python",
        type=str,
        help="path for the python environment to use to run hands pose estimation module",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    aws_bucket = config["bucket_s3"]

    video_dir = args.vid
    video_path = os.path.join(video_dir, "video", args.vid + ".mp4")

    body_pose_estimator = None
    if args.model == "face_mediapipe":
        body_pose_estimator = FaceMediapipeEstimator(config)

    body_pose_estimator.pose_estimation(
        video_path=video_path,
        write_on_pickle=True,
        pickle_filename="output.pkl",
        save_video_result=False,
        save_imgs_resulted=False,
        python_environment=args.python_environment,
    )
    os.remove(video_path)
    shutil.rmtree(video_dir, ignore_errors=True)
