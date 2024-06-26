import argparse
import distutils.util
import yaml
import os
import shutil
from estimators.body.hybrik_estimator import HybrIKEstimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", default=None, type=str, help="model to use for hands pose estimation"
    )
    parser.add_argument(
        "vid", default=None, type=str, help="vid of the video (save on aws S3)"
    )
    parser.add_argument(
        "--save_video_result",
        default="False",
        type=str,
        help="define if must be saved the video result with MANO mesh",
    )
    parser.add_argument(
        "--save_imgs_resulted",
        default="False",
        type=str,
        help="define if must be saved the single rendered images",
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

    body_pose_estimator = None
    if args.model == "hybrik":
        body_pose_estimator = HybrIKEstimator(config)

    body_pose_estimator.pose_estimation(
        video_path=args.vid,
        write_on_pickle=True,
        pickle_filename="output.pkl",
        save_video_result=bool(distutils.util.strtobool(args.save_video_result)),
        save_imgs_resulted=bool(distutils.util.strtobool(args.save_imgs_resulted)),
        python_environment=args.python_environment,
    )
    print(f"Done {args.vid}!")
