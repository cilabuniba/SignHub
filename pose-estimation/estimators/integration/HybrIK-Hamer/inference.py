import argparse
import os
import pickle
import shutil
import subprocess

import cv2
import numpy as np
import torch
from tqdm import tqdm

from estimators.body.HybrIK.hybrik.utils.render_pytorch3d import render_mesh
from integration import integrate

from estimators.body.HybrIK.hybrik.models.layers.smplx.body_models import SMPLXLayer
import pickle as pk

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

def get_kinematic_map(smplx_model, dst_idx):
    cur = dst_idx
    kine_map = dict()
    while cur >= 0:
        parent = int(smplx_model.parents[cur])
        if cur != dst_idx:  # skip the dst_idx itself
            kine_map[parent] = cur
        cur = parent
    return kine_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_folder", default=None, type=str,
                        help="video frames folder")
    parser.add_argument("--video_path", default=None, type=str,
                        help="video path")
    parser.add_argument("--hybrik_pickle_path", default=None, type=str,
                        help="output hybrik pickle path")
    parser.add_argument("--hamer_pickle_path", default=None, type=str,
                        help="output hamer pickle path")
    parser.add_argument("--integrated_folder", default="hybrik-hamer", type=str,
                        help="output hybrik + hamer folder")
    parser.add_argument("--integrated_output_pickle", default="hybrik-hamer/output.pkl", type=str,
                        help="output hybrik + hamer output pickle")
    parser.add_argument('--save_video_result', action='store_true', default=False,
                        help="define if must be saved the video result with SMPL-X mesh")
    parser.add_argument('--save_imgs_resulted', action='store_true', default=False,
                        help="define if must be saved the single rendered images")
    parser.add_argument('--save_video_result_mirrored', action='store_true', default=False,
                        help="define if must be saved the video result with SMPL-X mesh mirrored with "
                             "the original video")
    # parse args
    args = parser.parse_args()

    render_frames = args.save_video_result or args.save_imgs_resulted or args.save_video_result_mirrored

    with open(args.hybrik_pickle_path, mode='rb') as f:
        hybrik_data = pickle.load(f)

    with open(args.hamer_pickle_path, mode='rb') as f:
        hamer_data = pickle.load(f)

    if not os.path.exists(args.integrated_folder):
        os.makedirs(args.integrated_folder)

    frames_result_dir = os.path.join(args.integrated_folder, 'frames_result')
    if not os.path.exists(frames_result_dir) and (args.save_video_result or args.save_imgs_resulted):
        os.makedirs(frames_result_dir)

    frames_mirrored_result_dir = os.path.join(args.integrated_folder, 'frames_mirrored_result')
    if not os.path.exists(frames_mirrored_result_dir) and args.save_video_result_mirrored:
        os.makedirs(frames_mirrored_result_dir)

    _, info, _ = get_video_info(args.video_path)
    video_basename = os.path.basename(args.video_path).split('.')[0]

    write_stream = None
    if args.save_video_result:
        savepath = f'./{args.integrated_folder}/hybrik_hamer_integrated_result.mp4'
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

    hybrik_smplx_layer = SMPLXLayer(
        # model_path='model_files/smpl_v1.1.0/smplx/SMPLX_NEUTRAL.npz',
        model_path='estimators/body/HybrIK/model_files/smplx/SMPLX_NEUTRAL.npz',
        num_betas=10,
        use_pca=False,
        age='kid',
        kid_template_path='estimators/body/HybrIK/model_files/smplx_kid_template.npy',
    )
    left_hand_cinematic_map = get_kinematic_map(hybrik_smplx_layer, 20)
    right_hand_cinematic_map = get_kinematic_map(hybrik_smplx_layer, 21)
    shapedirs = torch.cat([hybrik_smplx_layer.shapedirs, hybrik_smplx_layer.expr_dirs], dim=-1).cuda()

    all_betas = hybrik_data["betas"]["body"]
    all_expression = hybrik_data["expression"]["body"]
    all_pose_skeleton = hybrik_data["joints"]["body"]
    all_rot_mats = hybrik_data["rot_mats"]["body"]
    all_phis = hybrik_data["phi"]["body"]
    all_transl = hybrik_data["camera"]
    all_focal = hybrik_data["focal"]

    all_left_hand_rot_mat = hamer_data["rot_mats"]["left_hand"]
    all_right_hand_rot_mat = hamer_data["rot_mats"]["right_hand"]
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

    smplx_faces = torch.from_numpy(hybrik_smplx_layer.faces.astype(np.int32))

    for idx, img_path in enumerate(tqdm(img_path_list)):
        dirname = os.path.dirname(img_path)
        basename = os.path.basename(img_path)
        betas = all_betas[idx].detach().cuda()
        expression = all_expression[idx].cuda()
        shape_components = torch.cat([betas, expression], dim=-1)

        left_hand_frame_pose = torch.from_numpy(all_left_hand_rot_mat[idx]["hand_pose"]).cuda()
        right_hand_frame_pose = torch.from_numpy(all_right_hand_rot_mat[idx]["hand_pose"]).cuda()
        left_hand_frame_global_orient = torch.from_numpy(all_left_hand_rot_mat[idx]["global_orient"]).cuda()
        right_hand_frame_global_orient = torch.from_numpy(all_right_hand_rot_mat[idx]["global_orient"]).cuda()

        rot_mats = all_rot_mats[idx].cuda()
        phis = all_phis[idx].cuda()
        vertices, joints, full_pose = integrate(
            shape_components, rot_mats, phis,
            hybrik_smplx_layer.v_template.cuda(), shapedirs, hybrik_smplx_layer.posedirs.cuda(),
            hybrik_smplx_layer.J_regressor.cuda(), hybrik_smplx_layer.extended_parents.clone(),
            hybrik_smplx_layer.children_map.clone().cuda(),
            hybrik_smplx_layer.lbs_weights.cuda(), train=hybrik_smplx_layer.training,
            leaf_indices=hybrik_smplx_layer.LEAF_INDICES, leaf_thetas=None,
            use_hand_pca=False,
            lhand_filter_matrix=hybrik_smplx_layer.lhand_filter_matrix,
            rhand_filter_matrix=hybrik_smplx_layer.rhand_filter_matrix,
            naive=False,
            left_hand_frame_pose=left_hand_frame_pose,
            right_hand_frame_pose=right_hand_frame_pose,
            left_hand_frame_global_orient=left_hand_frame_global_orient,
            right_hand_frame_global_orient=right_hand_frame_global_orient,
            left_hand_kinematic_map=left_hand_cinematic_map,
            right_hand_kinematic_map=right_hand_cinematic_map)

        transl = all_transl[idx]
        vertices = vertices.detach()
        focal = all_focal[idx]

        data_dict["rot_mats"]["body"].append(full_pose)
        data_dict["betas"]["body"].append(betas)
        data_dict["expression"]["body"].append(expression)
        data_dict["phi"]["body"].append(phis)
        data_dict["joints"]["body"].append(joints)
        data_dict["camera"].append(transl)
        data_dict["focal"].append(focal)

        # Visualization
        if render_frames:
            input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            image = input_image.copy()

            verts_batch = vertices
            transl_batch = transl

            color_batch = render_mesh(
                vertices=verts_batch, faces=smplx_faces,
                translation=transl_batch,
                focal_length=focal, height=image.shape[0], width=image.shape[1])

            if args.save_video_result or args.save_imgs_resulted:
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

                res_path = os.path.join(frames_result_dir, f'image-{idx+1:06d}.jpg')
                cv2.imwrite(res_path, image_vis)
                if args.save_video_result:
                    write_stream.write(image_vis)

            if args.save_video_result_mirrored:
                image_vis = color[:, :, :3]

                image_vis = image_vis.astype(np.uint8)
                image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

                height, width, _ = image.shape
                combined_width = width * 2
                combined_image = np.zeros((height, combined_width, 3), dtype=np.uint8)

                image = image.astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                combined_image[:, :width, :] = image
                combined_image[:, width:, :] = image_vis

                res_path = os.path.join(frames_mirrored_result_dir, f'image-{idx+1:06d}.jpg')
                cv2.imwrite(res_path, combined_image)

    with open(args.integrated_output_pickle, 'wb') as fid:
        pk.dump(data_dict, fid)

    if args.save_video_result:
        write_stream.release()

    if not args.save_imgs_resulted:
        shutil.rmtree(frames_result_dir, ignore_errors=True)

    if args.save_video_result_mirrored:
        mirrored_video_path = f'./{args.integrated_folder}/hybrik_hamer_integrated_mirrored_result.mp4'

        mirrored_img_seq = '{}/{}.jpg'.format(frames_mirrored_result_dir, 'image-%06d')
        cmd = ['ffmpeg', '-r', str(info['fps']), '-y', '-i', mirrored_img_seq,
               '-pix_fmt', 'yuv420p', mirrored_video_path]
        subprocess.call(cmd)

        shutil.rmtree(frames_mirrored_result_dir, ignore_errors=True)

