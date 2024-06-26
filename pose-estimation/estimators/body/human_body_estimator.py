from abc import ABC, abstractmethod

from estimators.human.human_estimator import HumanEstimator


class HumanBodyEstimator(HumanEstimator, ABC):

    def __init__(self, config):
        super().__init__(config)
        self.required_body_params = self.config["required_body_output_params"]

    @abstractmethod
    def set_pose_dir(self):
        raise NotImplementedError

    @abstractmethod
    def body_estimation_module(self, save_video_result, save_imgs_resulted, python_environment,
                               save_video_result_mirrored=None):
        """Specific pose estimation 3D for human body.
        :return dict rot_mat (required_hands_rot_mat_params), shape, camera (required_hpe_output_params) parameters"""
        pass

    def pose_estimation_module(self, save_video_result, save_imgs_resulted, python_environment,
                               save_video_result_mirrored=None):
        """Pose estimation 3D for human body.
        :return dict rot_mat (required_hands_rot_mat_params), shape, camera (required_hpe_output_params) parameters"""
        body_output = self.body_estimation_module(save_video_result, save_imgs_resulted, python_environment,
                                                  save_video_result_mirrored=save_video_result_mirrored)

        self.check_hpe_output(output=body_output, required_params=self.required_body_params)

        return body_output



