import numpy as np
import torch
import json
from torch.utils.data import DataLoader
import os
import numpy as np
from os.path import join as pjoin
from PN_Model import  AutoencoderPN, AutoencoderTransPN
from loss import ChamferDistance
from tqdm import tqdm
import pickle
import copy 
DEVICE = torch.device('cuda:0')
# DEVICE = torch.device('cpu')


def quaternion_to_matrix(quaternion):
    # Split the quaternion into x, y, z, and w components
    x, y, z, w = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]

    # Precompute products to optimize the matrix calculation
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Create the 3x3 rotation matrix
    rotation_matrix = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), torch.zeros_like(w),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), torch.zeros_like(w),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), torch.zeros_like(w)
    ], dim=-1).reshape(quaternion.shape[:-1] + (3, 4))

    bottom_row = torch.zeros(quaternion.shape[:-1] + (1, 4), device=quaternion.device)
    bottom_row[..., 0, 3] = 1

    transformation_matrix = torch.cat([rotation_matrix, bottom_row], dim=-2)

    return transformation_matrix

def apply_transformation(points, transformation_matrix):
    # points: [bs, 1024, 3]
    # transformation_matrix: [bs, 4, 4]
    
    bs, num_points, _ = points.shape
    ones = torch.ones((bs, num_points, 1), device=points.device)
    points_homogeneous = torch.cat([points, ones], dim=-1)  # [bs, 1024, 4]

    transformed_points_homogeneous = torch.bmm(points_homogeneous, transformation_matrix.transpose(1, 2))  # [bs, 1024, 4]

    transformed_points = transformed_points_homogeneous[..., :3]  # [bs, 1024, 3]

    return transformed_points


def eval(fps_pc_root_dir):
    feat_dim = 128 # 64 or 128
    model_type = 'PN' # ['PN', 'TransPN']
    scaled = False

    if scaled:
        exp_name = f'{model_type}_{feat_dim}_scaled'
    else:
        exp_name = f'{model_type}_{feat_dim}'
    ckpt_path = f'./utils/autoencoding_ours/ckpts/{exp_name}/029900.pth'

    root_dir = '/data0/v-wenbowang/Desktop/Logs/Final_Versions/full_train_best_0_zero_visual/results_trajectory'

    if model_type =='PN':
        model = AutoencoderPN(k=feat_dim, num_points=1024)
    else:
        model = AutoencoderTransPN(k=feat_dim, num_points=1024)

    ckpt_path = os.path.join(ckpt_path)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    loss_func = ChamferDistance()

    for obj_id in tqdm(range(3200), total=3200):

        folder_name = os.path.join(root_dir, f"{obj_id:04d}_seed0", "trajectory_small")
        out_folder_name = os.path.join(root_dir, f"{obj_id:04d}_seed0", exp_name) # trajectory feat output dir
        # os.makedirs(out_folder_name, exist_ok=True)
        obj_name_file = os.path.join(folder_name, 'env_object_scale.txt')
        l_count = 0
        with open(obj_name_file, 'r') as file:
            for line in file:
                if l_count > 0:
                    continue
                l_count+=1
        obj_pc_subpath = line.split('/')[:2]
        scale_str = line.split('\n')[0].split('/')[-1]
        ply_file = os.path.join(fps_pc_root_dir, obj_pc_subpath[0], obj_pc_subpath[1],'coacd', f'pc_fps1024_{scale_str}.npy')
        pc = np.load(ply_file, allow_pickle=True)
        pc_centered = pc - pc.mean(0)
        pc_centered = torch.tensor(pc_centered).float()


        # normalize to canonical space
        if not scaled:
            d = torch.sqrt((pc_centered**2).sum(1))
            pc_centered /= d.max()
            pass


        # for traj_id in tqdm(range(10), total=10):
        for traj_id in range(10):
            traj_path = os.path.join(folder_name, f"trajectory_small_{traj_id:03d}.pkl")
            out_traj_path = os.path.join(out_folder_name, f"trajectory_small_{traj_id:03d}.pkl")
            with open(traj_path, 'rb') as file:
                traj = pickle.load(file)
                # dict_keys(['observations', 'actions', 'values', 'successes', 'final_successes'])
                obs = traj['observations']
                quat = obs[:,:,194:198]
                traj_num, traj_len = obs.shape[:2]
                # pc_centered = torch.tensor(pc_centered).float()
                pc_centered_batch = copy.deepcopy(pc_centered)
                bs = quat.shape[0] * quat.shape[1]
                batch_pc = pc_centered_batch.unsqueeze(0).expand(bs, 1024, 3).float()
                quat = torch.tensor(quat).float()
                rot_mat = quaternion_to_matrix(quat) # 100, 200, 4, 4
                rot_mat = rot_mat.view(-1, 4, 4)
                rot_batch_pc =  apply_transformation(batch_pc, torch.tensor(rot_mat).float())

                running_error = 0.0
                mini_batch = 0

                feat_emb_tmp = []
                for mini_batch in range(traj_num):
                    with torch.no_grad():
                        x = rot_batch_pc[mini_batch*traj_len: (mini_batch+1)*traj_len].permute(0, 2, 1).cuda()
                        feat_emb, x_restored = model(x) # feat_emb: [BS, 64, 1], restored_pc: [BS, 3, 1024] 
                        feat_emb_tmp.append(feat_emb.detach().cpu())
                        # loss = loss_func(x, x_restored)
                        # running_error += loss.item()
                # print(f'Avg loss: {running_error/traj_num}')
                feat_emb_tmp = torch.concatenate(feat_emb_tmp)
                feat_emb_tmp = feat_emb_tmp.squeeze(-1)
                feat_emb_tmp = feat_emb_tmp.reshape(traj_num, traj_len, -1)
                saved_feat = feat_emb_tmp.numpy()

                # save traj
                with open(out_traj_path, 'wb') as f: 
                    pickle.dump(saved_feat, f)
                pass


if __name__=='__main__':
    fps_pc_root_dir = '/home/v-leizhou/zl_dev/AutoEncoder/Assets/meshdatav3_pc_fps'
    eval(fps_pc_root_dir)
