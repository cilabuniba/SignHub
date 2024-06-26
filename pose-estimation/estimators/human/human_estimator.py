import os.path
import pickle
import subprocess
from abc import ABC, abstractmethod

import yaml

import cv2

class HumanEstimator(ABC):

    def __init__(self, config):
        self.video_path = None
        self.vid = None
        self.frames_path = None
        self.video_fps = None
        self.config = config
        self.fps_crop = 30

    @abstractmethod
    def set_pose_dir(self):
        raise NotImplementedError


    def extract_frames_from_vid(self):
        """Extract frames from video."""
        subprocess.call(f'ffmpeg -i {self.video_path} -vf fps={self.fps_crop} {self.frames_path}/%06d.png', shell=True)

    @abstractmethod
    def pose_estimation_module(self, save_video_result, save_imgs_resulted, python_environment,
                               save_video_result_mirrored=None):
        """Pose estimation 3D single module for human body/hands/face.
        :return dict rot_mat, shape, camera (required_hpe_output_params) parameters"""
        pass

    def pose_estimation(self, video_path, write_on_pickle=True, pickle_filename="output.pkl", save_video_result=False,
                        save_imgs_resulted=False, python_environment="python", save_video_result_mirrored=None):
        """Pose estimation 3D for human body/hands/face.
        :return dict rot_mat, shape, camera (required_hpe_output_params) parameters"""
        self.set_current_video(video_path)
        self.set_pose_dir()
        result = self.pose_estimation_module(save_video_result, save_imgs_resulted, python_environment,
                                             save_video_result_mirrored)

        if write_on_pickle:
            output_pickle = os.path.join(self.pose_dir, pickle_filename)
            with open(output_pickle, 'wb') as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return result

    def set_current_video(self, video_path):
        self.video_path = video_path
        self.vid = os.path.splitext(os.path.basename(video_path))[0]
        self.frames_path = self.config["video_frames_path"].format(self.vid)
        os.makedirs(self.frames_path, exist_ok=True)
        video = cv2.VideoCapture(video_path)
        self.video_fps = video.get(cv2.CAP_PROP_FPS)

    def check_hpe_output(self, output, required_params):
        for key in required_params.keys():
            if key not in output.keys():
                raise Exception(f"{key} must be in output of the hpe module")
            if len(required_params[key]):
                for key_inner in required_params[key]:
                    if key_inner not in output[key].keys():
                        raise Exception(f"{key}:{key_inner} must be in output of the hpe module")