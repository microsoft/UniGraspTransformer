import torch
import pickle
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
 
 
class ObjectTrajectoryDataset(Dataset):
    """
        Object Trajectory Dataset: Nobj x Ntraj x {(Nstep, Nobs), (Nstep, Nact), (Nstep, Npoint, 4), (Nstep, 1), (1,)}
        Object Trajectory Dataset: 3200 x  1000 x {(  200,  300), (  200,   24), (  200,   2048, 4), (  200, 1), (1,)}
    """
    def __init__(self, trajectory_dir, target_object_lines, sample_ratio=1, sample_object=False, dtype=torch.float32, device='cuda:0'):
 
        # init dataset info
        self.dtype = dtype
        self.device = device
        self.sample_ratio = sample_ratio * 0.8
        self.sample_object = sample_object
        self.trajectory_dir = trajectory_dir
        self.target_object_lines = target_object_lines
 
        # get object number and trajectory number
        self.num_object = len(target_object_lines)
        self.num_trajectory = 1000  # per object
 
        # get trajectory group number
        self.group_size = 10  # per pickle
        # get sample length: total_trajectory // group_size
        self.sample_length = self.num_object * (self.num_trajectory // self.group_size)
 
    def __len__(self):
        # return sampled data_size: 3200 x 100
        return self.sample_length
   
    def __getitem__(self, idx):
        # locate object and trajectory file
        nobj = idx // (self.num_trajectory // self.group_size)
        ntraj = idx % (self.num_trajectory // self.group_size)
        # load object_trajectory data
        sample = self._load_object_trajectory(nobj, ntraj)
        return sample

    # load object trajectory data
    def _load_object_trajectory(self, nobj, ntraj):
        # load object_trajectory_data: {'rendered': (Ngroup, 200, Npoint*2, 4), 'canonical': (Ngroup, 200, Npoint, 3), 'valids': (Ngroup, 200, 1), 'final_successes': (Ngroup, 1, )}
        object_trajectory_data_path = osp.join(self.trajectory_dir, '{:04d}_seed0'.format(self.target_object_lines[nobj]), 'pointcloud/pointcloud_{:03d}.pkl'.format(ntraj))
        # [['rendered', (10, 200, 2048, 4)], ['canonical', (10, 200, 1024, 3)], ['hand', (10, 200, 36, 3)], ['successes', (10, 1)], ['final_successes', (10, 1)], ['valids', (10, 200, 1)]]
        object_trajectory_data = pickle.load(open(object_trajectory_data_path, "rb"))
        # delete useless items
        if 'canonical' in object_trajectory_data: object_trajectory_data.pop('canonical')

        # sample batch trajectory
        if self.sample_ratio != 1:
            # generate permute indices (Ngroup, 200)
            indices = np.random.permutation(np.arange(object_trajectory_data['rendered'].shape[1]))
            indices = indices[:int(indices.shape[-1] * self.sample_ratio)]
            indices[0] = 0
            # sample batch data
            for key, value in object_trajectory_data.items():
                if len(value.shape) > 2: object_trajectory_data[key] = value[:, indices]
        # sample object points
        if self.sample_object:
            # sample and center object points
            points = object_trajectory_data['rendered']
            points, appears = self.sample_label_points(points.reshape(-1, *points.shape[-2:]), label=3, number=1024)
            points = points[:, :, :3] - np.expand_dims(points[:, :, :3].mean(1), axis=1)
            object_trajectory_data['rendered'] = points.reshape(object_trajectory_data['rendered'].shape[0], object_trajectory_data['rendered'].shape[1], 1024, 3)
            # update valids with appear
            valids = object_trajectory_data['valids']
            valids = valids.reshape(-1, valids.shape[-1]) * np.expand_dims(appears, axis=1)
            object_trajectory_data['valids'] = valids.reshape(object_trajectory_data['valids'].shape[0], object_trajectory_data['valids'].shape[1], 1)
        else:
            object_trajectory_data['rendered'][..., 2] -= 0.6
        return object_trajectory_data

    def sample_label_points(self, points, label, number):
        # get nbatch, npoint, ndata
        nbatch, npoint, ndata = points.shape
        # get label_flags
        label_flags = points[..., -1] == label
        # init label_points
        label_points, appears = [], np.ones(nbatch)
    
        for n in range(nbatch):
            num_sample = np.sum(label_flags[n])
            # minus points with label
            if num_sample == 0:
                samples, appears[n] = np.zeros((number, ndata), dtype=points.dtype), 0
            # label points with label
            else:
                samples = points[n][label_flags[n]][np.random.randint(0, num_sample, size=number)]
            # append label_points
            label_points.append(samples)
        return np.stack(label_points), appears
    