import os
import pickle
import shutil
import subprocess
from abc import ABC

from estimators.hands.human_hands_estimator import HumanHandsEstimator


class HamerEstimator(HumanHandsEstimator, ABC):

    def __init__(self, config):
        super().__init__(config)
        self.pose_dir = None
        self.hamer_pickle_out_path = None

    def set_pose_dir(self):
        pose_dir = "pose/hamer"
        self.pose_dir = os.path.join(self.vid, pose_dir)
        self.hamer_pickle_out_path = os.path.join(self.pose_dir, "output_tmp.pkl")

    def hands_estimation_module(
        self,
        save_video_result,
        save_imgs_resulted,
        python_environment,
        save_video_result_mirrored=None,
    ):
        """Pose estimation 3D for human hands with Hamer.
        :return dict rot_mat (required_hands_rot_mat_params), shape, camera (required_hpe_output_params) parameters
        """

        self.extract_frames_from_vid()
        process_cmd = (
            f"{python_environment} estimators/hands/Hamer/inference.py --frames_folder {self.frames_path} "
            f"--hamer_folder {self.pose_dir} --hamer_output_pickle {self.hamer_pickle_out_path} "
            f"--fps {self.video_fps}"
        )
        if save_video_result:
            process_cmd += " --save_video_result"
        if save_imgs_resulted:
            process_cmd += " --save_imgs_resulted"

        subprocess.call(process_cmd, shell=True)

        with open(self.hamer_pickle_out_path, mode="rb") as f:
            output_data = pickle.load(f)

        os.remove(self.hamer_pickle_out_path)

        shutil.rmtree(
            os.path.dirname(os.path.dirname(self.frames_path)),
            ignore_errors=True,
        )
        return output_data
