import torch

from estimators.body.HybrIK.hybrik.models.layers.smplx.utils import Tensor
from estimators.body.HybrIK.hybrik.models.layers.smplx.lbs import (blend_shapes, vertices2joints,
                                                                   batch_inverse_kinematics_transform_naive,
                                                                   matrix_to_axis_angle, axis_angle_to_matrix,
                                                                   batch_rigid_transform)
import torch.nn.functional as F

def sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def transfer_rot(body_pose_rotmat, part_rotmat, kinematic_map, transfer_type):

    rotmat= body_pose_rotmat[0]
    parent_id = 0
    while parent_id in kinematic_map:
        child_id = kinematic_map[parent_id]
        local_rotmat = body_pose_rotmat[child_id]
        rotmat = torch.matmul(rotmat, local_rotmat)
        parent_id = child_id

    if transfer_type == 'g2l':
        part_rot_new = torch.matmul(rotmat.T, part_rotmat)
    else:
        assert transfer_type == 'l2g'
        part_rot_new = torch.matmul(rotmat, part_rotmat)

    return part_rot_new

def integrate(
    betas: Tensor,
    rot_mats: Tensor,
    phis: Tensor,
    v_template: Tensor,
    shapedirs: Tensor,
    posedirs: Tensor,
    J_regressor: Tensor,
    parents: Tensor,
    children: Tensor,
    lbs_weights: Tensor,
    leaf_indices: Tensor,
    leaf_thetas: Tensor = None,
    train: bool = True,
    use_hand_pca: bool = False,
    lhand_filter_matrix: Tensor = None,
    rhand_filter_matrix: Tensor = None,
    naive=False,
    left_hand_frame_pose=None,
    right_hand_frame_pose=None,
    left_hand_frame_global_orient=None,
    right_hand_frame_global_orient=None,
    left_hand_kinematic_map=None,
    right_hand_kinematic_map=None,
):

    # parents should add leaf joints
    batch_size = max(betas.shape[0], rot_mats.shape[0])
    device, dtype = betas.device, betas.dtype
    num_theta = phis.shape[1] + 1

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    if leaf_thetas is not None:
        rest_J = vertices2joints(J_regressor, v_shaped)
    else:
        rest_J_inner = vertices2joints(J_regressor, v_shaped)

        leaf_vertices = v_shaped[:, leaf_indices].clone()
        rest_J = torch.cat([rest_J_inner, leaf_vertices], dim=1)

    # 3. Get the rotation matrics

    if use_hand_pca:
        rot_aa = matrix_to_axis_angle(rot_mats.reshape(batch_size, -1, 3, 3))
        lhand_aa = rot_aa[:, 25:40].reshape(batch_size, 45)
        rhand_aa = rot_aa[:, 40:55].reshape(batch_size, 45)

        lhand_aa = torch.einsum('bi,ij->bj', [lhand_aa, lhand_filter_matrix])
        rhand_aa = torch.einsum('bi,ij->bj', [rhand_aa, rhand_filter_matrix])
        lhand_rotmat = axis_angle_to_matrix(lhand_aa.reshape(batch_size, 15, 3))
        rhand_rotmat = axis_angle_to_matrix(rhand_aa.reshape(batch_size, 15, 3))

        rot_mats[:, 25:40] = lhand_rotmat
        rot_mats[:, 40:55] = rhand_rotmat

    if left_hand_frame_global_orient is not None:
        quaternion = matrix_to_quaternion(left_hand_frame_global_orient)
        axis_angle = quaternion_to_axis_angle(quaternion)
        axis_angle[0, 0, 1::3] *= -1
        axis_angle[0, 0, 2::3] *= -1
        quaternion = axis_angle_to_quaternion(axis_angle)
        rot_matrix = quaternion_to_matrix(quaternion).cuda()
        left_hand_frame_global_orient_local = transfer_rot(
            rot_mats[0], rot_matrix[0], left_hand_kinematic_map, "g2l")
        rot_mats[:, 20] = left_hand_frame_global_orient_local

    if right_hand_frame_global_orient is not None:
        right_hand_frame_global_orient_local = transfer_rot(
            rot_mats[0], right_hand_frame_global_orient[0].cuda(), right_hand_kinematic_map, "g2l")
        rot_mats[:, 21] = right_hand_frame_global_orient_local

    if left_hand_frame_pose is not None:
        quaternion = matrix_to_quaternion(left_hand_frame_pose)
        axis_angle = quaternion_to_axis_angle(quaternion)
        axis_angle[0, :, 1::3] *= -1
        axis_angle[0, :, 2::3] *= -1
        quaternion = axis_angle_to_quaternion(axis_angle)
        rot_matrix = quaternion_to_matrix(quaternion).cuda()
        rot_mats[:, 25:40] = rot_matrix[:, :]

    if right_hand_frame_pose is not None:
        rot_mats[:, 40:55] = right_hand_frame_pose[:, :]

    J_transformed, A = batch_rigid_transform(
        rot_mats, rest_J[:, :num_theta].clone(), parents[:num_theta], dtype=dtype)

    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
    pose_offsets = torch.matmul(pose_feature, posedirs) \
        .view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped.detach()

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    #root align
    root_j = J_transformed[:, [0], :].clone()
    J_transformed -= root_j
    verts -= root_j

    return verts, J_transformed, rot_mats