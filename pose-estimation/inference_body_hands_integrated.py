import argparse
import distutils.util
import os.path
import shutil

import yaml

from estimators.integration.hybrik_hamer_integrated_estimator import (
    HybrIKHamerIntegratedEstimator,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        default=None,
        type=str,
        help="model to use for body + hands pose estimation",
    )
    parser.add_argument(
        "vid", default=None, type=str, help="vid of the video (save on aws S3)"
    )
    parser.add_argument(
        "--save_video_result",
        default="False",
        type=str,
        help="define if must be saved the video result with SMPL-X mesh",
    )
    parser.add_argument(
        "--save_imgs_resulted",
        default="False",
        type=str,
        help="define if must be saved the single rendered images",
    )
    parser.add_argument(
        "--save_video_result_mirrored",
        default="False",
        type=str,
        help="define if must be saved the video result with SMPL-X mesh mirrored with "
        "the original video",
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

    video_dir = args.vid
    video_path = os.path.join(video_dir, "video", args.vid + ".mp4")

    body_pose_estimator = None
    hpe_module_output_files_to_download = []
    if args.model == "hybrik-hamer":
        hpe_module_output_files_to_download.append(
            os.path.join(video_dir, "pose", "hybrik", "output.pkl")
        )
        hpe_module_output_files_to_download.append(
            os.path.join(video_dir, "pose", "hamer", "output.pkl")
        )

        body_pose_estimator = HybrIKHamerIntegratedEstimator(config)

    body_pose_estimator.pose_estimation(
        video_path=video_path,
        write_on_pickle=True,
        pickle_filename="output.pkl",
        save_video_result=bool(distutils.util.strtobool(args.save_video_result)),
        save_imgs_resulted=bool(distutils.util.strtobool(args.save_imgs_resulted)),
        save_video_result_mirrored=bool(
            distutils.util.strtobool(args.save_video_result_mirrored)
        ),
        python_environment=args.python_environment,
    )
    os.remove(video_path)
    for file_to_download in hpe_module_output_files_to_download:
        os.remove(file_to_download)
    shutil.rmtree(video_dir, ignore_errors=True)
