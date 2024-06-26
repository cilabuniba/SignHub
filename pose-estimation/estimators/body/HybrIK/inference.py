"""Image demo script."""
import argparse
import os
import shutil

import sys

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLX
# from hybrik.utils.render_pytorch3d import render_mesh
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
import pickle as pk

det_transform = T.Compose([T.ToTensor()])

halpe_wrist_ids = [94, 115]
halpe_left_hand_ids = [
    5, 6, 7,
    9, 10, 11,
    17, 18, 19,
    13, 14, 15,
    1, 2, 3,
]
halpe_right_hand_ids = [
    5, 6, 7,
    9, 10, 11,
    17, 18, 19,
    13, 14, 15,
    1, 2, 3,
]

halpe_lhand_leaves = [
    8, 12, 20, 16, 4
]
halpe_rhand_leaves = [
    8, 12, 20, 16, 4
]


halpe_hand_ids = [i + 94 for i in halpe_left_hand_ids] + [i + 115 for i in halpe_right_hand_ids]
halpe_hand_leaves_ids = [i + 94 for i in halpe_lhand_leaves] + [i + 115 for i in halpe_rhand_leaves]


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def get_video_info(in_file):
    stream = cv2.VideoCapture(in_file)
    assert stream.isOpened(), 'Cannot capture source'
    # self.path = input_source
    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
    fps = stream.get(cv2.CAP_PROP_FPS)
    frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # bitrate = int(stream.get(cv2.CAP_PROP_BITRATE))
    videoinfo = {'fourcc': fourcc, 'fps': fps, 'frameSize': frameSize}
    stream.release()

    return stream, videoinfo, datalen


def recognize_video_ext(ext=''):
    if ext == 'mp4':
        return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
    elif ext == 'avi':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    elif ext == 'mov':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    else:
        print("Unknow video format {}, will use .mp4 instead of it".format(ext))
        return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'


def integral_hm(hms):
    # hms: [B, K, H, W]
    B, K, H, W = hms.shape
    hms = hms.sigmoid()
    hms = hms.reshape(B, K, -1)
    hms = hms / hms.sum(dim=2, keepdim=True)
    hms = hms.reshape(B, K, H, W)

    hm_x = hms.sum((2,))
    hm_y = hms.sum((3,))

    w_x = torch.arange(hms.shape[3]).to(hms.device).float()
    w_y = torch.arange(hms.shape[2]).to(hms.device).float()

    hm_x = hm_x * w_x
    hm_y = hm_y * w_y

    coord_x = hm_x.sum(dim=2, keepdim=True)
    coord_y = hm_y.sum(dim=2, keepdim=True)

    coord_x = coord_x / float(hms.shape[3]) - 0.5
    coord_y = coord_y / float(hms.shape[2]) - 0.5

    coord_uv = torch.cat((coord_x, coord_y), dim=2)
    return coord_uv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_folder", default=None, type=str,
                        help="video frames folder")
    parser.add_argument("--video_path", default=None, type=str,
                        help="video path")
    parser.add_argument("--hybrik_folder", default="hybrik", type=str,
                        help="output hamer folder")
    parser.add_argument("--hybrik_output_pickle", default="hybrik/output.pkl", type=str,
                        help="output hamer output pickle")
    parser.add_argument('--save_video_result', action='store_true', default=False,
                        help="define if must be saved the video result with MANO mesh")
    parser.add_argument('--save_imgs_resulted', action='store_true', default=False,
                        help="define if must be saved the single rendered images")

    # parse args
    args = parser.parse_args()

    render_frames = args.save_video_result or args.save_imgs_resulted

    cfg_file = 'estimators/body/HybrIK/configs/smplx/256x192_hrnet_rle_smplx_kid.yaml'
    CKPT = 'estimators/body/HybrIK/pretrained_models/hybrikx_rle_hrnet.pth'

    cfg = update_config(cfg_file)

    cfg['MODEL']['EXTRA']['USE_KID'] = cfg['DATASET'].get('USE_KID', False)
    cfg['LOSS']['ELEMENTS']['USE_KID'] = cfg['DATASET'].get('USE_KID', False)

    bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
    bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
    dummpy_set = edict({
        'joint_pairs_17': None,
        'joint_pairs_24': None,
        'joint_pairs_29': None,
        'bbox_3d_shape': bbox_3d_shape
    })

    data_dict = {
        "rot_mats": {
            "body": [],
        },
        "betas": {
            "body": [],
        },
        "expression": {
            "body": [],
        },
        "phi": {
            "body": [],
        },
        "joints": {
            "body": [],
        },
        "camera": [],
        "focal": [],
    }

    transformation = SimpleTransform3DSMPLX(
        dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
        color_factor=cfg.DATASET.COLOR_FACTOR,
        occlusion=cfg.DATASET.OCCLUSION,
        input_size=cfg.MODEL.IMAGE_SIZE,
        output_size=cfg.MODEL.HEATMAP_SIZE,
        depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
        bbox_3d_shape=bbox_3d_shape,
        rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
        train=False, add_dpg=False,
        loss_type=cfg.LOSS['TYPE'])

    det_model = fasterrcnn_resnet50_fpn(pretrained=True)

    hybrik_model = builder.build_sppe(cfg.MODEL)

    print(f'Loading model from {CKPT}...')
    save_dict = torch.load(CKPT, map_location='cpu')
    if type(save_dict) == dict:
        model_dict = save_dict['model']
        hybrik_model.load_state_dict(model_dict)
    else:
        hybrik_model.load_state_dict(save_dict)

    det_model.cuda(0)
    hybrik_model.cuda(0)
    det_model.eval()
    hybrik_model.eval()

    print('### Extract Image...')

    if not os.path.exists(args.hybrik_folder):
        os.makedirs(args.hybrik_folder)

    frames_result_dir = os.path.join(args.hybrik_folder, 'frames_result')
    if not os.path.exists(frames_result_dir) and render_frames:
        os.makedirs(frames_result_dir)

    _, info, _ = get_video_info(args.video_path)
    video_basename = os.path.basename(args.video_path).split('.')[0]

    write_stream = None
    if args.save_video_result:
        savepath = f'./{args.hybrik_folder}/hybrik_result.mp4'
        info['savepath'] = savepath

        write_stream = cv2.VideoWriter(
            *[info[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
        if not write_stream.isOpened():
            print("Try to use other video encoders...")
            ext = info['savepath'].split('.')[-1]
            fourcc, _ext = recognize_video_ext(ext)
            info['fourcc'] = fourcc
            info['savepath'] = info['savepath'][:-4] + _ext
            write_stream = cv2.VideoWriter(
                *[info[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])

        assert write_stream.isOpened(), 'Cannot open video for writing'

    files = os.listdir(args.frames_folder)
    files.sort()

    img_path_list = []

    for file in tqdm(files):
        if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:
            img_path = os.path.join(args.frames_folder, file)
            img_path_list.append(img_path)

    prev_box = None
    renderer = None
    smplx_faces = torch.from_numpy(hybrik_model.smplx_layer.faces.astype(np.int32))

    print('### Run Model...')
    idx = 0
    for img_path in tqdm(img_path_list):
        dirname = os.path.dirname(img_path)
        basename = os.path.basename(img_path)

        with torch.no_grad():
            # Run Detection
            input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            det_input = det_transform(input_image).to(0)
            det_output = det_model([det_input])[0]

            if prev_box is None:
                tight_bbox = get_one_box(det_output)  # xyxy
                if tight_bbox is None:
                    continue
            else:
                tight_bbox = get_one_box(det_output)  # xyxy

            if tight_bbox is None:
                tight_bbox = prev_box

            prev_box = tight_bbox

            # Run HybrIK
            # bbox: [x1, y1, x2, y2]
            pose_input, bbox, img_center = transformation.test_transform(
                input_image.copy(), tight_bbox)
            pose_input = pose_input.to(0)[None, :, :, :]

            # vis 2d
            bbox_xywh = xyxy2xywh(bbox)

            pose_output = hybrik_model(
                pose_input, flip_test=True,
                bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
                img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float(),
            )

            transl = pose_output.transl.detach()
            vertices = pose_output.pred_vertices.detach()
            #camera = pose_output.pred_camera
            focal = 1000.0
            bbox_xywh = xyxy2xywh(bbox)

            focal = focal / 256 * bbox_xywh[2]
            # Visualization
            if render_frames:
                image = input_image.copy()

                verts_batch = vertices
                transl_batch = transl

                color_batch = render_mesh(
                    vertices=verts_batch, faces=smplx_faces,
                    translation=transl_batch,
                    focal_length=focal, height=image.shape[0], width=image.shape[1])

                valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
                image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
                image_vis_batch = (image_vis_batch * 255).cpu().numpy()

                color = image_vis_batch[0]
                valid_mask = valid_mask_batch[0].cpu().numpy()
                input_img = image
                alpha = 0.9
                image_vis = alpha * color[:, :, :3] * valid_mask + (
                    1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img

                image_vis = image_vis.astype(np.uint8)
                image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

                idx += 1
                res_path = os.path.join(frames_result_dir, f'image-{idx:06d}.jpg')
                cv2.imwrite(res_path, image_vis)
                if args.save_video_result:
                    write_stream.write(image_vis)

            assert pose_input.shape[0] == 1, 'Only support single batch inference for now'

            pred_xyz_hybrik = pose_output.pred_xyz_hybrik.reshape(
                71, 3).cpu().data.numpy()

            data_dict["joints"]["body"].append(pred_xyz_hybrik)
            data_dict["rot_mats"]["body"].append(pose_output.rot_mats)
            data_dict["betas"]["body"].append(pose_output.pred_beta)
            data_dict["expression"]["body"].append(pose_output.pred_expression)
            data_dict["phi"]["body"].append(pose_output.pred_phi)
            data_dict["camera"].append(transl)
            data_dict["focal"].append(focal)

    with open(args.hybrik_output_pickle, 'wb') as fid:
        pk.dump(data_dict, fid)

    if args.save_video_result:
        write_stream.release()

    if not args.save_imgs_resulted:
        shutil.rmtree(frames_result_dir, ignore_errors=True)