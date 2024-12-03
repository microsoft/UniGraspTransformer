# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import torch
import numpy as np
import os.path as osp
from isaacgym.torch_utils import *

sys.path.append(osp.realpath(osp.join(osp.realpath(__file__), '../../../..')))


@torch.jit.script
def compute_heading_and_up(
    torso_rotation, inv_start_rot, to_target, vec0, vec1, up_idx
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    num_envs = torso_rotation.shape[0]
    target_dirs = normalize(to_target)

    torso_quat = quat_mul(torso_rotation, inv_start_rot)
    up_vec = get_basis_vector(torso_quat, vec1).view(num_envs, 3)
    heading_vec = get_basis_vector(torso_quat, vec0).view(num_envs, 3)
    up_proj = up_vec[:, up_idx]
    heading_proj = torch.bmm(heading_vec.view(
        num_envs, 1, 3), target_dirs.view(num_envs, 3, 1)).view(num_envs)

    return torso_quat, up_proj, heading_proj, up_vec, heading_vec


@torch.jit.script
def compute_rot(torso_quat, velocity, ang_velocity, targets, torso_positions):
    vel_loc = quat_rotate_inverse(torso_quat, velocity)
    angvel_loc = quat_rotate_inverse(torso_quat, ang_velocity)

    roll, pitch, yaw = get_euler_xyz(torso_quat)

    walk_target_angle = torch.atan2(targets[:, 2] - torso_positions[:, 2],
                                    targets[:, 0] - torso_positions[:, 0])
    angle_to_target = walk_target_angle - yaw

    return vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target


@torch.jit.script
def quat_axis(q, axis=0):
    # type: (Tensor, int) -> Tensor
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


@torch.jit.script
def batch_quat_apply(a, b):
    # unsqueeze a(Nenv, 1, 4)
    shape = b.shape
    a = a.unsqueeze(1)
    # extract the xyz component of quaternion a
    xyz = a[:, :, :3]
    # compute the cross product t
    t = torch.cross(xyz, b, dim=-1) * 2
    # compute the final result and reshape it to the original shape
    return (b + a[:, :, 3:] * t + torch.cross(xyz, t, dim=-1)).view(shape)

@torch.jit.script
# compute sided distance from sources(Nenv, Ns, 3) to targets(Nenv, Nt, 3)
def batch_sided_distance(sources, targets):
    # pairwise_distances: (Nenv, Ns, Nt)
    pairwise_distances = torch.cdist(sources, targets)
    # find the minimum distances
    distances, _ = torch.min(pairwise_distances, dim=-1)
    return distances

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot


# compute quat from current hand to current object pca axes
def compute_hand_to_object_pca_quat(object_pcas, object_rot, hand_rot):
    # Get rotated object PCA
    object_pca_rotates = batch_quat_apply(object_rot, object_pcas)
    # Get main object PCA
    object_pca_targets = object_pca_rotates[:, 0, :]
    # Get target vector normal to the pca axis 
    object_pca_targets[:, [0, 1]] = object_pca_targets[:, [1, 0]]
    object_pca_targets[:, -1] = 0
    object_pca_targets[:, 0] *= -1
    object_pca_targets[object_pca_targets[:, 1] > 0, :] *= -1
    object_pca_targets = object_pca_targets / object_pca_targets.norm(dim=1, keepdim=True)

    # Get hand_axes
    hand_axes = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(object_pca_targets.shape[0], 1).to(object_pca_targets.device)
    hand_axes = batch_quat_apply(hand_rot, hand_axes.unsqueeze(1)).squeeze(1)
    # Compute the quaternion from hand_axes to target_axes
    axis = torch.cross(hand_axes, object_pca_targets)
    cos_theta = torch.sum(hand_axes * object_pca_targets, dim=1) 
    sin_theta = torch.norm(hand_axes, dim=1)
    theta = torch.atan2(sin_theta, cos_theta)
    axis = axis / torch.norm(axis, dim=1, keepdim=True)
    # Ensure the quaternions are unit quaternions
    quaternions = torch.cat([axis * torch.sin(theta / 2).unsqueeze(1), torch.cos(theta / 2).unsqueeze(1)], dim=1)
    quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)
    # merge quaternions with hand_rot
    return object_pca_rotates, quat_mul(quaternions, hand_rot)


# compute hand_bodies: hand_joints (17) + hand_bodies (19)
def compute_hand_body_pos(hand_joint_pos, hand_joint_rot):
    # get device and num_envs
    device = hand_joint_pos.device
    num_envs = hand_joint_pos.shape[0]
    # compute hand body position from base and joint
    hand_body_pos = []
    for n in range(hand_joint_rot.shape[1]):
        if n in [2, 5, 8, 12]: continue
        elif n == 0:
            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], to_torch([1, 0, 0], device=device).repeat(num_envs, 1) * 0.03) \
                + quat_apply(hand_joint_rot[:, n, :],to_torch([0, 1, 0], device=device).repeat(num_envs, 1) * -0.005)
            hand_body_pos.append(body_pos)
            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], to_torch([-1, 0, 0], device=device).repeat(num_envs, 1) * 0.03) \
                + quat_apply(hand_joint_rot[:, n, :],to_torch([0, 1, 0], device=device).repeat(num_envs, 1) * -0.005)
            hand_body_pos.append(body_pos)
            
            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], to_torch([0, 0, 1], device=device).repeat(num_envs, 1) * 0.03) \
                + quat_apply(hand_joint_rot[:, n, :],to_torch([0, 1, 0], device=device).repeat(num_envs, 1) * -0.005)
            hand_body_pos.append(body_pos)
            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], to_torch([0, 0, 1], device=device).repeat(num_envs, 1) * 0.06) \
                + quat_apply(hand_joint_rot[:, n, :],to_torch([0, 1, 0], device=device).repeat(num_envs, 1) * -0.005)
            hand_body_pos.append(body_pos)

            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], to_torch([1, 0, 0], device=device).repeat(num_envs, 1) * 0.03) \
                + quat_apply(hand_joint_rot[:, n, :], to_torch([0, 0, 1], device=device).repeat(num_envs, 1) * 0.06) \
                + quat_apply(hand_joint_rot[:, n, :],to_torch([0, 1, 0], device=device).repeat(num_envs, 1) * -0.005)
            hand_body_pos.append(body_pos)

            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], to_torch([1, 0, 0], device=device).repeat(num_envs, 1) * 0.015) \
                + quat_apply(hand_joint_rot[:, n, :], to_torch([0, 0, 1], device=device).repeat(num_envs, 1) * 0.015) \
                + quat_apply(hand_joint_rot[:, n, :],to_torch([0, 1, 0], device=device).repeat(num_envs, 1) * -0.005)
            hand_body_pos.append(body_pos)

            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], to_torch([-1, 0, 0], device=device).repeat(num_envs, 1) * 0.03) \
                + quat_apply(hand_joint_rot[:, n, :], to_torch([0, 0, 1], device=device).repeat(num_envs, 1) * 0.03) \
                + quat_apply(hand_joint_rot[:, n, :],to_torch([0, 1, 0], device=device).repeat(num_envs, 1) * -0.005)
            hand_body_pos.append(body_pos)

        elif n == 10:
            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], to_torch([0, 0, 1], device=device).repeat(num_envs, 1) * 0.02) \
                + quat_apply(hand_joint_rot[:, n, :], to_torch([-1, 0, 0], device=device).repeat(num_envs, 1) * 0.015)
            hand_body_pos.append(body_pos)
        else:
            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], to_torch([0, 0, 1], device=device).repeat(num_envs, 1) * 0.02)
            hand_body_pos.append(body_pos)
    
    hand_body_pos = torch.stack(hand_body_pos, dim=1)
    hand_body_pos = torch.cat([hand_body_pos, hand_joint_pos], dim=1)
    return hand_body_pos


# compute hand_bodies: hand_joints (13) + hand_bodies (5)
def compute_inspire_hand_body_pos(hand_joint_pos, hand_joint_rot):
    # get device and num_envs
    device = hand_joint_pos.device
    num_envs = hand_joint_pos.shape[0]
    # compute hand body position from base and joint
    # {'R_index_intermediate_joint': 1, 'R_index_proximal_joint': 0, 'R_middle_intermediate_joint': 3, 'R_middle_proximal_joint': 2, 'R_pinky_intermediate_joint': 5, 'R_pinky_proximal_joint': 4, 
    # 'R_ring_intermediate_joint': 7, 'R_ring_proximal_joint': 6, 'R_thumb_distal_joint': 11, 'R_thumb_intermediate_joint': 10, 'R_thumb_proximal_pitch_joint': 9, 'R_thumb_proximal_yaw_joint': 8}
    hand_body_pos = []
    for n in range(hand_joint_rot.shape[1]):
        # self.fingertips = ['R_index_intermediate', 'R_middle_intermediate', 'R_ring_intermediate', 'R_pinky_intermediate', 'R_thumb_distal']
        if n in [2, 4, 8, 6]:
            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], to_torch([0, 1, 0], device=device).repeat(num_envs, 1) * 0.02)
            hand_body_pos.append(body_pos)
        elif n == 0:
            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], to_torch([0, 0, 1], device=device).repeat(num_envs, 1) * 0.10) + quat_apply(hand_joint_rot[:, n, :], to_torch([0, 1, 0], device=device).repeat(num_envs, 1) * -0.02)
            hand_body_pos.append(body_pos)
    hand_body_pos = torch.stack(hand_body_pos, dim=1)
    hand_body_pos = torch.cat([hand_body_pos, hand_joint_pos], dim=1)
    return hand_body_pos