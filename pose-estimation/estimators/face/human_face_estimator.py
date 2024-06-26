from abc import ABC, abstractmethod

from estimators.human.human_estimator import HumanEstimator


class HumanFaceEstimator(HumanEstimator, ABC):

    def __init__(self, config):
        super().__init__(config)
        self.required_face_params = self.config["required_face_output_params"]

    @abstractmethod
    def set_pose_dir(self):
        raise NotImplementedError

    @abstractmethod
    def face_estimation_module(self, save_video_result, save_imgs_resulted, python_environment,
                                save_video_result_mirrored=None):
        """Specific pose estimation 3D for human face blendshapes.
        :return dict of face blendshapes"""
        pass

    def pose_estimation_module(self, save_video_result, save_imgs_resulted, python_environment,
                               save_video_result_mirrored=None):
        """Pose estimation 3D for human face blendshapes.
        :return dict of face blendshapes"""
        blendshapes_output = self.face_estimation_module(save_video_result, save_imgs_resulted, python_environment,
                                                    save_video_result_mirrored=save_video_result_mirrored)

        self.check_hpe_output(output=blendshapes_output, required_params=self.required_face_params)

        return blendshapes_output
