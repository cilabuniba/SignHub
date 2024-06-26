import argparse

import os
import pickle

import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default=None, type=str,
                        help="video path")
    parser.add_argument("--face_mediapipe_folder", default="face_mediapipe", type=str,
                        help="output face mediapipe folder")
    parser.add_argument("--face_mediapipe_output_pickle", default="face_mediapipe/output.pkl", type=str,
                        help="output face mediapipe output pickle")

    args = parser.parse_args()

    os.makedirs(args.face_mediapipe_folder, exist_ok=True)

    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path='estimators/face/Face_Mediapipe/model/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1,
                                           running_mode=VisionRunningMode.VIDEO)

    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    data_dict = {
        "blendshapes": []
    }
    previous_face_blendshapes = None
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            count += 1

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            frame_timestamp_ms = int(count / fps * 1000)
            face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            face_blendshapes = {}
            if not len(face_landmarker_result.face_blendshapes):
                data_dict["blendshapes"].append(previous_face_blendshapes)
                continue
            for blendshape in face_landmarker_result.face_blendshapes[0]:
                face_blendshapes[blendshape.category_name] = blendshape.score
            data_dict["blendshapes"].append(face_blendshapes)
            previous_face_blendshapes = face_blendshapes

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    with open(args.face_mediapipe_output_pickle, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)