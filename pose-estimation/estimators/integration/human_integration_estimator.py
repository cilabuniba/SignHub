from abc import ABC, abstractmethod

from estimators.human.human_estimator import HumanEstimator


class HumanIntegrationEstimator(HumanEstimator, ABC):

    def __init__(self, config):
        super().__init__(config)
        self.required_integration_params = self.config["required_body_output_params"]

    @abstractmethod
    def set_pose_dir(self):
        raise NotImplementedError

    @abstractmethod
    def integration_estimation_module(self, save_video_result, save_imgs_resulted, python_environment,
                                      save_video_result_mirrored=None):
        """Specific pose estimation 3D for human body +hands.
        :return dict rot_mat (required_hands_rot_mat_params), shape, camera (required_hpe_output_params) parameters"""
        pass

    def pose_estimation_module(self, save_video_result, save_imgs_resulted, python_environment,
                               save_video_result_mirrored=None):
        """Pose estimation 3D for human body +hands.
        :return dict rot_mat (required_hands_rot_mat_params), shape, camera (required_hpe_output_params) parameters"""
        integrated_output = self.integration_estimation_module(save_video_result, save_imgs_resulted,
                                                               python_environment,
                                                               save_video_result_mirrored=save_video_result_mirrored)

        self.check_hpe_output(output=integrated_output, required_params=self.required_integration_params)

        return integrated_output
