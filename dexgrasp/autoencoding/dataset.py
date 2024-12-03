import os
import torch
import numpy as np
from plyfile import PlyData
from torch.utils.data import Dataset
from scipy.stats import ortho_group
import glob

import numpy as np
import torch
import os
import os.path as osp
import open3d as o3d
import time
import yaml

class PointCloudsDex(Dataset):

    def __init__(self, dataset_path, random_rotate):
        """
        Arguments:
            is_training: a boolean.
        """


        split_path = ('./utils/autoencoding_ours/splits/train_set_results.txt') # 3200 objs

        use_scale_pair = True
        self.random_rotate = random_rotate
        split_folder_list = []
        obj_name_list = []
        object_code_list = []
        obj_scale_list = []
        with open(split_path, 'r') as file:
            for line in file:
                obj_info = line.strip().split('\'')[1]
                split_folder = obj_info.split('/')[0]
                obj_name = obj_info.split('/')[1]
                obj_name_list.append(obj_name)
                split_folder_list.append(split_folder)
                object_code_list.append(obj_info)
                obj_scale_list.append(line.split(':')[1].split(',')[0][1:-1])
        obj_num = len(object_code_list)
        num_pts = 1024 
        scale2str = {
            0.06: '006',
            0.08: '008',
            0.10: '010',
            0.12: '012',
            0.15: '015',
        }
        file_name_list = ['pc_fps1024_012.npy', 'pc_fps1024_008.npy', 'pc_fps1024_006.npy', 'pc_fps1024_010.npy', 'pc_fps1024_015.npy']
        scale_to_str_dict = {
            '0.06':'pc_fps1024_006.npy',
            '0.08':'pc_fps1024_008.npy',
            '0.1': 'pc_fps1024_010.npy',
            '0.12':'pc_fps1024_012.npy',
            '0.15':'pc_fps1024_015.npy',
        }
        self.paths = []
        for i, object_code in enumerate(object_code_list):
            obj_ply_dir = dataset_path + '/' + object_code
            obj_ply_dir = os.path.join(dataset_path, object_code, 'coacd')
            scale_str = obj_scale_list[i]
            file_name =scale_to_str_dict[scale_str]
            npy_dir = osp.join(obj_ply_dir, file_name)
            self.paths.append((npy_dir, i))




    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        """
        Returns:
            x: a float tensor with shape [3, num_points].
        """
        
        npy_dir, _ = self.paths[i]
        
        with open(npy_dir, 'rb') as f:
            pts = np.load(f)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3])

        x = np.asarray(pcd.points)
        
        x -= x.mean(0) # centerize
        d = np.sqrt((x ** 2).sum(1)) 
        x /= d.max() # normalize to canonical space
        if self.random_rotate:
            x = augmentation(x) # only random rotation
        
        x = torch.FloatTensor(x).permute(1, 0)
        return x

def load_ply(filename):
    """
    Arguments:
        filename: a string.
    Returns:
        a float numpy array with shape [num_points, 3].
    """
    ply_data = PlyData.read(filename)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    return points.astype('float32')


from scipy.stats import ortho_group

def augmentation(x):
    """
    Arguments:
        x: a float numpy array with shape [b, n, 3].
    Returns:
        a float numpy array with shape [b, n, 3].
    """
    # batch size
    b = x.shape[0]

    # random rotation matrix
    m = ortho_group.rvs(3)  # shape [b, 3, 3]
    m = np.expand_dims(m, 0)  # shape [b, 1, 3, 3]
    m = m.astype('float32')

    x = np.expand_dims(x, 1)
    x = np.matmul(x, m)
    x = np.squeeze(x, 1)

    return x
