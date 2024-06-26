import os
import pickle
import shutil
import subprocess
from abc import ABC

from estimators.face.human_face_estimator import HumanFaceEstimator


class FaceMediapipeEstimator(HumanFaceEstimator, ABC):

    def __init__(self, config):
        super().__init__(config)
        self.pose_dir = None
        self.face_mediapipe_pickle_out_path = None

    def set_pose_dir(self):
        pose_dir = "pose/face_mediapipe"
        self.pose_dir = os.path.join(self.vid, pose_dir)
        self.face_mediapipe_pickle_out_path = os.path.join(self.pose_dir, "output_tmp.pkl")

    def face_estimation_module(self, save_video_result, save_imgs_resulted, python_environment,
                               save_video_result_mirrored=None):
        """Specific pose estimation 3D for human face blendshapes.
        :return dict of face blendshapes"""

        process_cmd = (f"{python_environment} estimators/face/Face_Mediapipe/inference.py "
                       f"--video_path {self.video_path} "
                       f"--face_mediapipe_folder {self.pose_dir} "
                       f"--face_mediapipe_output_pickle {self.face_mediapipe_pickle_out_path}")

        subprocess.call(process_cmd, shell=True)

        with open(self.face_mediapipe_pickle_out_path, mode='rb') as f:
            output_data = pickle.load(f)

        os.remove(self.face_mediapipe_pickle_out_path)

        return output_data
