import os
import time
import tqdm
import cv2 as cv
import numpy as np
from utils.general_utils import *
from torch.utils.data import Dataset


class ObjectTrajectoryDatasetBatch(Dataset):
    """
        Object Trajectory Dataset: Nobj x Ntraj x {(Nstep, Nobs), (Nstep, Nact), (Nstep, 1), (1,)}
        Object Trajectory Dataset: 3200 x  1000 x {(  200,  300), (  200,   24), (  200, 1), (1,)}
    """
    def __init__(self, config, log_dir, asset_dir, trajectory_dir, object_scale_yaml, target_object_lines, dtype=torch.float32, device='cuda:0'):

        # init dataset info
        self.dtype = dtype
        self.device = device
        self.config = config
        self.log_dir = log_dir
        self.asset_dir = asset_dir
        self.trajectory_dir = trajectory_dir
        self.target_object_lines = target_object_lines
        # locate object mesh and feature folder
        self.object_mesh_dir = osp.join(self.asset_dir, 'meshdatav3_scaled')
        self.visual_feat_dir = osp.join(self.asset_dir, 'meshdatav3_pc_feat')
        # load object_scale_yaml
        self.object_line_list, self.object_scale_list, self.object_scale_dict = load_object_scale_result_yaml(object_scale_yaml)
        # self.object_scale_dict = {object_code: [scale], }
        # self.object_scale_list = [object_code/scale, ]
        # self.object_line_list = [object_line, ]

        # init valid object scales
        self.scale2str = {0.06: '006', 0.08: '008', 0.10: '010', 0.12: '012', 0.15: '015'}
        self.str2scale = {'006': 0.06, '008': 0.08, '010': 0.10, '012': 0.12, '015': 0.15}

        # get object number and trajectory number
        self.num_object = len(target_object_lines)
        self.num_trajectory = self.config['Offlines']['num_trajectory'] if 'num_trajectory' in self.config['Offlines'] else 1000

        # get train_epochs and train_batchs
        self.train_epochs = self.config['Offlines']['train_epochs']  # 20 / 50
        self.train_batchs = self.config['Offlines']['train_batchs']  # 100 / 200
        # get train_iterations
        self.train_iterations = self.num_object * self.num_trajectory // self.train_batchs
        # set log_iterations
        self.log_times = 1 if self.num_object == 1 else 10
        self.log_iterations = self.train_iterations // self.log_times

        # set trajectory group number
        self.group_size = 10
        # get sample length: total_trajectory // group_size
        self.sample_length = self.num_object * (self.num_trajectory // self.group_size)

        # set loading hyper params
        self.load_values = False
        # load dynamic object visual features
        self.load_dynamic_visual_feats = True if 'dynamic_object_visual_feature' in self.config['Offlines'] and self.config['Offlines']['dynamic_object_visual_feature'] else False
        # load static object visual feature
        self.load_static_visual_feats = not self.config['Offlines']['zero_object_visual_feature'] and not self.load_dynamic_visual_feats
        # load static_object_visual_feats (Nobj, 64)
        self._load_static_object_visual_feats()

        # save config yaml
        save_yaml(os.path.join(self.log_dir, 'train.yaml'), self.config)

        # vision_based training: use_external_feature, update observations
        self.vision_based = True if 'vision_based' in self.config['Modes'] and self.config['Modes']['vision_based'] else False

        # replace Obs with Obs_Dataset: load saved observation from dataset
        if 'Obs_Dataset' in self.config: self.config['Obs'] = self.config['Obs_Dataset']
        # use_external_feature, locate external_feature name and size
        self.use_external_feature = True if 'use_external_feature' in self.config['Offlines'] and self.config['Offlines']['use_external_feature'] else False
        self.external_feature_name = self.config['Offlines']['external_feature_name'] if self.use_external_feature else None
        # self.external_feature_size = int(self.external_feature_name.split('_')[-1]) if self.use_external_feature else None


    def __len__(self):
        # return sampled data_size
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
        # load object_trajectory_data: {'observations': (Ngroup, 200, Nobs), 'actions': (Ngroup, 200, Nact), 'features': (Ngroup, 200, 64), 'values': (Ngroup, 200, 1), 'valids': (Ngroup, 200, 1), 'successes: (Ngroup, 1, )'}
        object_trajectory_data_path = osp.join(self.trajectory_dir, '{:04d}_seed0'.format(self.target_object_lines[nobj]), 'trajectory/trajectory_{:03d}.pkl'.format(ntraj))
        object_trajectory_data = load_pickle(object_trajectory_data_path)
        
        # load dynamic_object_visual_feature
        if self.load_dynamic_visual_feats:
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['object_visual'][0]:self.config['Obs']['intervals']['object_visual'][1]] = object_trajectory_data['features']
        # load static_object_visual_feats
        if self.load_static_visual_feats:
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['object_visual'][0]:self.config['Obs']['intervals']['object_visual'][1]] = 0.1 * self.static_object_visual_feats[nobj, :]
        
        # vision_based: update observations
        if self.vision_based:
            # load rendered object_state: features, centers, hand_object
            object_state_path = osp.join(self.trajectory_dir, '{:04d}_seed0'.format(self.target_object_lines[nobj]), 'pointcloud/pointcloud_{:03d}.pkl'.format(ntraj))
            object_state = load_pickle(object_state_path)
            # check valid appears within trajectory
            object_state = check_object_valid_appears(object_state['valids'], object_state)
            # update object features
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['object_visual'][0]:self.config['Obs']['intervals']['object_visual'][1]] = object_state['features']
            # update object centers
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]:self.config['Obs']['intervals']['objects'][1]] *= 0
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]:self.config['Obs']['intervals']['objects'][0]+3] = object_state['centers']
            # update hand_objects
            object_trajectory_data['observations'][..., self.config['Obs']['intervals']['hand_objects'][0]:self.config['Obs']['intervals']['hand_objects'][1]] = object_state['hand_object']
            # update valids with appears
            object_trajectory_data['valids'] *= object_state['appears']
            # use object pcas, estimated from rendered object points
            if 'use_object_pcas' in self.config['Offlines'] and self.config['Offlines']['use_object_pcas']:
                # get object pcas
                object_pcas = object_state['pcas'].reshape(object_state['pcas'].shape[0], object_state['pcas'].shape[1], -1)
                # use dynamic or static object pcas
                if self.config['Offlines']['use_dynamic_pcas']: object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]+6:self.config['Obs']['intervals']['objects'][0]+15] = object_pcas
                else: object_trajectory_data['observations'][..., self.config['Obs']['intervals']['objects'][0]+6:self.config['Obs']['intervals']['objects'][0]+15] = object_pcas[:, 0, :][:, None, :]

        # # send object_trajectory_data to GPU tensor
        # for key, value in object_trajectory_data.items():
        #     object_trajectory_data[key] = torch.tensor(object_trajectory_data[key], dtype=self.dtype, device=self.device)
        return object_trajectory_data


    # load static_object_visual_feats (Nobj, 64)
    def _load_static_object_visual_feats(self):
        # init target_object_visual_feats
        self.static_object_visual_feats = np.zeros((self.num_object, 64))
        # load visual_features for target_object_lines
        for nline, line in enumerate(self.target_object_lines):
            # locate object_scale visual_feature
            split_temp = self.object_scale_list[line].split('/')
            object_code, scale_str = '{}/{}'.format(split_temp[0], split_temp[1]), self.scale2str[float(split_temp[2])] 
            file_dir = osp.join(self.visual_feat_dir, '{}/pc_feat_{}.npy'.format(object_code, scale_str))
            # load object_scale visual_feature
            with open(file_dir, 'rb') as file: self.static_object_visual_feats[nline, :] = np.load(file)