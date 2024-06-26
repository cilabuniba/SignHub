import os
import pickle
import shutil
import subprocess
from abc import ABC

from estimators.integration.human_integration_estimator import HumanIntegrationEstimator


class HybrIKHamerIntegratedEstimator(HumanIntegrationEstimator, ABC):

    def __init__(self, config):
        super().__init__(config)
        self.pose_dir = None
        self.hybrik_pickle_out_path = None
        self.hamer_pickle_out_path = None
        self.integrated_pickle_out_path = None

    def set_pose_dir(self):
        pose_dir = "pose/hybrik-hamer"
        hybrik_pose_dir = "pose/hybrik"
        hamer_pose_dir = "pose/hamer"
        self.pose_dir = os.path.join(self.vid, pose_dir)
        self.integrated_pickle_out_path = os.path.join(self.pose_dir, "output_tmp.pkl")
        self.hybrik_pickle_out_path = os.path.join(self.vid, hybrik_pose_dir, "output.pkl")
        self.hamer_pickle_out_path = os.path.join(self.vid, hamer_pose_dir, "output.pkl")

    def integration_estimation_module(self, save_video_result, save_imgs_resulted, python_environment,
                                      save_video_result_mirrored=None):
        """Pose estimation 3D for human body with HybrIK.
        :return dict rot_mat (required_hands_rot_mat_params), shape, camera (required_hpe_output_params) parameters"""
        self.extract_frames_from_vid()

        process_cmd = (f"{python_environment} estimators/integration/HybrIK-Hamer/inference.py "
                       f"--frames_folder {self.frames_path} --video_path {self.video_path} "
                       f"--hybrik_pickle_path {self.hybrik_pickle_out_path} "
                       f"--hamer_pickle_path {self.hamer_pickle_out_path} "
                       f"--integrated_folder {self.pose_dir} "
                       f"--integrated_output_pickle {self.integrated_pickle_out_path} ")
        if save_video_result:
            process_cmd += " --save_video_result"
        if save_imgs_resulted:
            process_cmd += " --save_imgs_resulted"
        if save_video_result_mirrored:
            process_cmd += " --save_video_result_mirrored"

        subprocess.call(process_cmd, shell=True)

        with open(self.integrated_pickle_out_path, mode='rb') as f:
            output_data = pickle.load(f)

        os.remove(self.integrated_pickle_out_path)

        shutil.rmtree(self.frames_path, ignore_errors=True)
        return output_data

