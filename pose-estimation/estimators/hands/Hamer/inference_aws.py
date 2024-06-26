import argparse
import os

from demo_aws import hands_pose_estimation_hamer
from utils import config
from utils.utils_s3 import download_one_file, multiup

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vid", default=None, type=str, help="video id (name of the S3 directory)")

    # parse args
    args = parser.parse_args()
    if args.vid == None:
        args.vid = "test-application"
    video_directory = f"{args.vid}"
    video_path = f"{video_directory}/video/{args.vid}.mp4"
    # Download video file
    print(f"Downloading video {video_path} from S3...")
    download_one_file(s3_file_path=video_path, aws_bucket=config.cfg["bucket-s3"])
    assert os.path.isfile(video_path)
    if os.path.isfile(video_path):
        print(f"File {video_path} download correctly")

    # process video file
    print(f"Estimating human pose 3D using Hamer...")
    output_pose_path = f"{video_directory}/pose/"
    hands_pose_estimation_hamer(video_path=video_path, pkl_output_dir=output_pose_path)

    # upload files on S3
    print(f"Uploading output pose on S3 {output_pose_path}...")
    multiup(files_path=output_pose_path, aws_bucket=config.cfg["bucket-s3"])