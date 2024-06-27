import pickle
import shutil
import subprocess
from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.abspath("estimators/hands/Hamer/detectron2"))

from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import hamer

#TODO to implement save_video_result: if param is True must be generated a video with all frames collected
# in a sequence using ffmpeg. And if save_imgs_resulted after this operation the frames must be removed
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_folder", default=None, type=str,
                        help="video frames folder")
    parser.add_argument("--hamer_folder", default="hamer", type=str,
                        help="output hamer folder")
    parser.add_argument("--hamer_output_pickle", default="hamer/output.pkl", type=str,
                        help="output hamer output pickle")
    parser.add_argument('--save_video_result', action='store_true', default=False,
                        help="define if must be saved the video result with MANO mesh")
    parser.add_argument('--save_imgs_resulted', action='store_true', default=False,
                        help="define if must be saved the single rendered images")
    parser.add_argument('--fps', default=30,
                        help="video fps")

    # parse args
    args = parser.parse_args()

    render_frames = args.save_video_result or args.save_imgs_resulted

    checkpoint = DEFAULT_CHECKPOINT
    os.makedirs(args.hamer_folder, exist_ok=True)

    file_type = ['*.jpg', '*.png']

    # Download and load checkpoints
    model, model_cfg = load_hamer(checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector

    # cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    # detectron2_cfg = LazyConfig.load(str(cfg_path))
    # detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    # for i in range(3):
    #     detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    # detector = DefaultPredictor_Lazy(detectron2_cfg)

    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    img_folder_resulted = None
    renderer = None
    if render_frames:
        img_folder_resulted = os.path.join(args.hamer_folder, "frames_result")
        os.makedirs(img_folder_resulted, exist_ok=True)
        renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Get all demo images ends with .jpg or .png
    files = os.listdir(args.frames_folder)
    files.sort()

    img_paths = []

    for file in tqdm(files):
        if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:
            img_path = os.path.join(args.frames_folder, file)
            img_paths.append(img_path)

    data_dict = {
        "rot_mats": {
            "left_hand": [],
            "right_hand": []
        },
        "joints": {
            "left_hand": [],
            "right_hand": []
        },
        "camera": [],
    }
    previous_left_hand = np.zeros((21, 3))
    previous_right_hand = np.zeros((21, 3))

    previous_left_hand_params = None
    previous_right_hand_params = None

    previous_camera = None

    # Iterate over all images in folder
    for idx, img_path in enumerate(img_paths):
        print(f"IDX: {idx} - IMG PATH: {img_path}")

        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img_cv2,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            data_dict["joints"]["left_hand"].append(previous_left_hand)
            data_dict["rot_mats"]["left_hand"].append(previous_left_hand_params)
            data_dict["joints"]["right_hand"].append(previous_right_hand)
            data_dict["rot_mats"]["right_hand"].append(previous_right_hand_params)
            data_dict["camera"].append(previous_camera)

            continue

        elif len(bboxes) == 1 and is_right[0] == 0:
            data_dict["joints"]["right_hand"].append(previous_right_hand)
            data_dict["rot_mats"]["right_hand"].append(previous_right_hand_params)

        elif len(bboxes) == 1 and is_right[0] == 1:
            data_dict["joints"]["left_hand"].append(previous_left_hand)
            data_dict["rot_mats"]["left_hand"].append(previous_left_hand_params)

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2 * batch['right'] - 1)
            # scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            scaled_focal_length = 250 / model_cfg.MODEL.IMAGE_SIZE * img_size.max()

            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                               scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                if render_frames:
                    # Get filename from path img_path
                    img_fn, _ = os.path.splitext(os.path.basename(img_path))
                    person_id = int(batch['personid'][n])
                    white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:, None, None] / 255) / (
                                DEFAULT_STD[:, None, None] / 255)
                    input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:, None, None] / 255) + (
                                DEFAULT_MEAN[:, None, None] / 255)
                    input_patch = input_patch.permute(1, 2, 0).numpy()

                    regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                              out['pred_cam_t'][n].detach().cpu().numpy(),
                                              batch['img'][n],
                                              mesh_base_color=LIGHT_BLUE,
                                              scene_bg_color=(1, 1, 1),
                                              )

                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                    cv2.imwrite(os.path.join(img_folder_resulted, f'{img_fn}_{person_id}.png'), 255 * final_img[:, :, ::-1])

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                keypoints_3d = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                keypoints_3d[:, 0] = (2 * is_right - 1) * keypoints_3d[:, 0]
                pred_mano_params = out['pred_mano_params']
                if is_right == 0:
                    data_dict["joints"]["left_hand"].append(keypoints_3d)
                    data_dict["rot_mats"]["left_hand"].append(pred_mano_params)
                    previous_left_hand = keypoints_3d
                    previous_left_hand_params = pred_mano_params

                else:
                    data_dict["joints"]["right_hand"].append(keypoints_3d)
                    data_dict["rot_mats"]["right_hand"].append(pred_mano_params)

                    previous_right_hand = keypoints_3d
                    previous_right_hand_params = pred_mano_params

                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

                data_dict["camera"].append(cam_t)
                previous_camera = cam_t

        # Render front view
        if len(all_verts) > 0 and render_frames:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n],
                                                     is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)  # Add alpha channel
            input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

            cv2.imwrite(os.path.join(img_folder_resulted, f'{img_fn}.jpg'), 255 * input_img_overlay[:, :, ::-1])

    if args.save_video_result:
        img_seq = '{}/{}.jpg'.format(img_folder_resulted, '%06d')
        video_path = os.path.join(args.hamer_folder, "hamer_result.mp4")
        cmd = ['ffmpeg', '-r', str(args.fps), '-y', '-i', img_seq, '-pix_fmt', 'yuv420p', '-c:v', 'libx264',
               '-crf', '23', video_path]
        subprocess.call(cmd)
    if not args.save_imgs_resulted:
        shutil.rmtree(img_folder_resulted, ignore_errors=True)

    with open(args.hamer_output_pickle, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
