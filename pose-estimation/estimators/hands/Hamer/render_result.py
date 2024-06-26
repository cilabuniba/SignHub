import os
import pickle
import time

import numpy as np
import torch
from matplotlib import pyplot as plt, animation
from textwrap import wrap

from scipy.signal import savgol_filter
from torch.nn.functional import normalize


HANDS = "left"  # left | right | both

assert HANDS == "left" or HANDS == "right" or HANDS == "both"

dir_vec_pairs_first_hand = [
    (0, 1, 0.15),
    (1, 2, 0.05),
    (2, 3, 0.03),
    (3, 4, 0.01),

    (0, 5, 0.2),
    (5, 6, 0.1),
    (6, 7, 0.05),
    (7, 8, 0.02),

    (0, 9, 0.25),
    (9, 10, 0.15),
    (10, 11, 0.08),
    (11, 12, 0.03),

    (0, 13, 0.2),
    (13, 14, 0.1),
    (14, 15, 0.05),
    (15, 16, 0.02),

    (0, 17, 0.1),
    (17, 18, 0.05),
    (18, 19, 0.03),
    (19, 20, 0.01),
]

dir_vec_pairs_second_hand = [
    (21, 22, 0.4),
    (22, 23, 0.2),
    (23, 24, 0.1),
    (24, 25, 0.05),

    (21, 26, 0.4),
    (26, 27, 0.2),
    (27, 28, 0.1),
    (28, 29, 0.05),

    (21, 30, 0.4),
    (30, 31, 0.2),
    (31, 32, 0.1),
    (32, 33, 0.05),

    (21, 34, 0.4),
    (34, 35, 0.2),
    (35, 36, 0.1),
    (36, 37, 0.05),

    (21, 38, 0.4),
    (38, 39, 0.2),
    (39, 40, 0.1),
    (40, 41, 0.05),
]

dir_vec_pairs_both_hands = dir_vec_pairs_first_hand + dir_vec_pairs_second_hand

if HANDS == "left" or HANDS == "right":
    dir_vec_pairs = dir_vec_pairs_first_hand
else:
    dir_vec_pairs = dir_vec_pairs_both_hands

def convert_dir_vec_to_pose(vec):
    # vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 3:
        joint_pos = torch.zeros((vec.shape[0], 49, 3)).to("cpu")
        for j, pair in enumerate(dir_vec_pairs):
            for frame in range(joint_pos.shape[0]):
                if all(vec[frame, j]) == 0:
                    joint_pos[frame, pair[1]] = torch.zeros(3).to("cpu")
                else:
                    joint_pos[frame, pair[1]] = (joint_pos[frame, pair[0]] + torch.Tensor([pair[2]]).to("cpu")
                                                 * vec[frame, j])
    else:
        assert False

    return joint_pos

def create_video_and_save(save_path, output, title, aux_str=None):
    print('Rendering video...')
    start = time.time()

    fig = plt.figure(figsize=(8, 4))
    axes = [fig.add_subplot(1, 2, 1, projection='3d'), fig.add_subplot(1, 2, 2, projection='3d')]
    axes[0].view_init(elev=20, azim=-60)
    axes[1].view_init(elev=20, azim=-60)
    fig_title = title

    if aux_str:
        fig_title += ('\n' + aux_str)
    fig.suptitle('\n'.join(wrap(fig_title, 75)), fontsize='medium')

    output_poses = convert_dir_vec_to_pose(torch.Tensor(output).to("cpu"))

    def animate(i):
        colors = [
            'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
            'sienna', 'teal', 'navy', 'maroon', 'coral', 'lime', 'indigo', 'gold', 'orchid', 'salmon',
            'tomato', 'darkgreen', 'peru', 'steelblue', 'darkorange', 'mediumseagreen', 'royalblue', 'mediumvioletred',
            'dodgerblue', 'chartreuse',
            'slategray', 'darkmagenta', 'darkslategray', 'orangered', 'mediumaquamarine', 'mediumslateblue',
            'saddlebrown', 'indianred', 'cadetblue', 'darkgoldenrod',
            'seagreen', 'mediumblue', 'firebrick', 'darkorchid', 'limegreen', 'cornflowerblue', 'chocolate', 'crimson',
            'darkturquoise', 'darkkhaki',
            'mediumspringgreen', 'slateblue', 'rosybrown', 'mediumorchid', 'mediumturquoise', 'darkcyan',
            'palevioletred', 'greenyellow', 'darkolivegreen', 'darkred',
            'deepskyblue', 'springgreen', 'midnightblue', 'sienna', 'hotpink', 'darkviolet', 'darkslateblue',
            'cadetblue', 'darkslategray', 'darkorange'
        ]
        pose = output_poses[i].cpu()
        if pose is not None:
            axes[0].clear()
            for j, pair in enumerate(dir_vec_pairs):
                color = colors[j]
                axes[0].plot([pose[pair[0], 0], pose[pair[1], 0]],
                             [pose[pair[0], 2], pose[pair[1], 2]],
                             [pose[pair[0], 1], pose[pair[1], 1]],
                             zdir='z', linewidth=1.5, color=color)
                #print(f"Directional Vector {j}: {color}")
            axes[0].set_xlim3d(-0.5, 0.5)
            axes[0].set_ylim3d(0.5, -0.5)
            axes[0].set_zlim3d(0.5, -0.5)
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('z')
            axes[0].set_zlabel('y')
            axes[0].set_title('{} ({}/{})'.format("generated", i + 1, len(output_poses)))

    num_frames = len(output_poses)
    ani = animation.FuncAnimation(fig, animate, interval=1, frames=num_frames, repeat=False)

    # save video
    try:
        ani.save(save_path, fps=60, dpi=150)  # dpi 150 for a higher resolution
        del ani
        plt.close(fig)
    except RuntimeError:
        assert False, 'RuntimeError'

    print('done, took {:.1f} seconds'.format(time.time() - start))
    return output_poses

def convert_pose_seq_to_dir_vec(pose):
    pose = torch.from_numpy(pose).float()
    if pose.shape[-1] != 3:
        pose = pose.reshape(pose.shape[:-1] + (-1, 3))

    if len(pose.shape) == 3:
        dir_vec = torch.zeros((pose.shape[0], len(dir_vec_pairs), 3))
        for i, pair in enumerate(dir_vec_pairs):
            for frame in range(pose.shape[0]):
                if all(pose[frame, pair[1]]) == 0 or all(pose[frame, pair[0]]) == 0:
                    continue
                else:
                    dir_vec[frame, i] = pose[frame, pair[1]] - pose[frame, pair[0]]

            dir_vec[:, i, :] = normalize(dir_vec[:, i, :], dim=1)  # to unit length
    else:
        assert False

    return dir_vec

def smooth_dir_vec(dir_vec, window_length=5, polyorder=3):
    # Parametri del filtro di Savitzky-Golay
    # Dimensione della finestra: deve essere un intero dispari
    # Ordine del polinomio da adattare alla finestra: deve essere un intero minore o uguale a window_length

    # Reshape directional_vectors a una forma temporanea di dimensione (N, K, 3)
    temp_vectors = dir_vec.reshape((dir_vec.shape[0], -1, 3))

    # Applica il filtro di Savitzky-Golay lungo la coordinata temporale per ciascun vettore
    smoothed_dir_vec = np.stack(
        [savgol_filter(temp_vectors[:, i, :], window_length, polyorder, axis=0) for i in range(temp_vectors.shape[1])],
        axis=1)

    # Ritorna alla forma originale Nx(Kx3)
    smoothed_dir_vec = smoothed_dir_vec.reshape(dir_vec.shape)
    return smoothed_dir_vec

def rotate_around_x(joints, angle_degrees):
    # Converti l'angolo da gradi a radianti
    angle_radians = np.radians(angle_degrees)

    # Crea la matrice di rotazione attorno all'asse x
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_radians), -np.sin(angle_radians)],
        [0, np.sin(angle_radians), np.cos(angle_radians)]
    ])

    # Applica la rotazione a ciascun punto/joint
    rotated_joints = np.dot(joints, rotation_matrix.T)

    return rotated_joints

if __name__ == '__main__':

    with open("./output/hamer_hands_jts.pkl", 'rb') as file:
        data_jts_hands = pickle.load(file)  # HANDS POSE: HAMER
        data_jts_left_hand = data_jts_hands["left_hand_jts"]
        data_jts_right_hand = data_jts_hands["right_hand_jts"]

    #data_jts_left_hand = rotate_around_x(data_jts_left_hand, 180)

    if HANDS == "left":
        output_filename = "rendering_left_hand.mp4"
    elif HANDS == "right":
        output_filename = "rendering_right_hand.mp4"
    else:
        output_filename = "rendering_both_hands.mp4"

    output_path = os.path.join('./output', output_filename)
    poses = []
    for left_hand_jts, right_hand_jts in zip(data_jts_left_hand, data_jts_right_hand):

        frame_jtx_skeleton = []
        if HANDS == "left" or HANDS == "both":
            for left_jts in left_hand_jts:
                frame_jtx_skeleton += list(left_jts)
        if HANDS == "right" or HANDS == "both":
            for right_jts in right_hand_jts:
                frame_jtx_skeleton += list(right_jts)

        poses.append(frame_jtx_skeleton)

    poses = np.asarray(poses).reshape(len(poses), -1)
    dir_vec = convert_pose_seq_to_dir_vec(poses)
    dir_vec = np.asarray(dir_vec)
    dir_vec = dir_vec.reshape(dir_vec.shape[0], -1)
    dir_vec = smooth_dir_vec(dir_vec, 13, 3)
    create_video_and_save(save_path=output_path, output=dir_vec,
                          title="TEST HAMER KEYPOINTS")