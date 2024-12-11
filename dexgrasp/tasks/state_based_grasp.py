# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os.path as osp
import os, glob, tqdm
import random, torch, trimesh

from isaacgym import gymtorch
from isaacgym import gymapi

from utils.general_utils import *
from utils.torch_jit_utils import *
from utils.hand_model import ShadowHandModel
from utils.render_utils import PytorchBatchRenderer

from sklearn.decomposition import PCA
from tasks.hand_base.base_task import BaseTask

sys.path.append(osp.join(BASE_DIR, 'dexgrasp/autoencoding'))
from autoencoding.PN_Model import AutoencoderPN, AutoencoderTransPN


# StateBasedGrasp task, includes env setup, simulate step
class StateBasedGrasp(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):

        # init task setting
        self.cfg = cfg
        self.sim_params = sim_params
        self.agent_index = agent_index
        self.physics_engine = physics_engine
        self.is_multi_agent = is_multi_agent
        # load train/test config: modes and weights
        self.algo = cfg['algo']
        self.config = cfg['config']
        # vision_based setting
        self.vision_based = True if 'vision_based' in self.config['Modes'] and self.config['Modes']['vision_based'] else False
        if self.vision_based: self.cfg["env"]["numEnvs"] = min(10, self.cfg["env"]["numEnvs"])  # limit to 10 environments to increase speed
        # init vision_based_tracker 
        self.vision_based_tracker = None
        # init params from cfg
        self.init_wandb = self.cfg["wandb"]
        self.object_scale_file = self.cfg["object_scale_file"]
        self.start_line, self.end_line, self.group = self.cfg["start_line"], self.cfg["end_line"], self.cfg["group"]
        self.shuffle_dict, self.shuffle_env = self.cfg['shuffle_dict'], self.cfg['shuffle_env']
        # init params from cfg['task']
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        # init params from cfg['env']
        # # Run params
        self.up_axis = 'z'
        self.num_envs = self.cfg["env"]["numEnvs"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.is_testing, self.test_epoch, self.test_iteration, self.current_test_iteration = cfg['test'], cfg['test_epoch'], self.cfg["test_iteration"], 0
        self.current_iteration = 0
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        # # Reward params
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        # # Control params
        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]
        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations
        # # Reset params
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]
        # # Success params
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        # print("Averaging factor: ", self.av_factor)

        # # Control frequency
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)
        
        # # Observation params
        self.obs_type = self.cfg["env"]["observationType"]
        # print("Obs type:", self.obs_type)
        
        # image size
        self.image_size = 256
        # table size
        self.table_dims = gymapi.Vec3(1, 1, 0.6)
        self.table_center = np.array([0.0, 0.0, self.table_dims.z])

        # full-state observation
        num_obs = 236 + 64
        self.num_obs_dict = {"full_state": num_obs}
        # observation space
        self.num_hand_obs = 66 + 95 + 24 + 6  # 191 =  22*3 + (65+30) + 24
        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal"]
        self.hand_center = ["robot0:palm"]
        self.num_fingertips = len(self.fingertips) 
        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
        num_states = 0
        if self.asymmetric_obs: num_states = 211
        self.cfg["env"]["numStates"] = num_states
        self.num_agents = 1
        self.cfg["env"]["numActions"] = 24 
        # # Device params
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["graphics_device_id"] = device_id
        # # Visualize params
        self.cfg["headless"] = headless
        # self.debug_viz = False if headless else True
        
        # # Render settings
        self.render_each_view = self.cfg["render_each_view"]
        self.render_hyper_view = self.cfg["render_hyper_view"]
        self.render_point_clouds = self.cfg["render_point_clouds"]
        self.sample_point_clouds = self.cfg["sample_point_clouds"]
        # init render folder and render_env_list
        self.render_folder = None
        self.render_env_list = list(range(9)) if self.render_hyper_view else None
        self.render_env_list = list(range(self.num_envs)) if self.sample_point_clouds else self.render_env_list

        # # Camera params
        self.frame = -1
        self.num_cameras = CAMERA_PARAMS['num'] if (self.render_each_view or self.sample_point_clouds) else 0  # render all views
        self.num_cameras = 1 if self.render_hyper_view else self.num_cameras  # render hyper view only
        # init camera infos
        self.camera_handle_list = []
        self.camera_depth_tensor_list, self.camera_rgb_tensor_list, self.camera_seg_tensor_list = [], [], []
        self.camera_view_mat_list, self.camera_vinv_mat_list, self.camera_proj_mat_list = [], [], []
        # create camera configs
        self.create_cfg_cameras()
        
        # default init from BaseTask create gym, sim, viewer, buffers for obs and states
        super().__init__(cfg=self.cfg, enable_camera_sensors=True if (headless or self.render_each_view or self.render_hyper_view or self.sample_point_clouds) else False)

        # set viewer camera pose
        self.look_at_env = None
        if self.viewer != None:
            cam_pos = gymapi.Vec3(2.5, 0, 2)
            cam_target = gymapi.Vec3(0, 0, 0)
            self.look_at_env = self.envs[len(self.envs) // 2]
            self.gym.viewer_camera_look_at(self.viewer, self.look_at_env, cam_pos, cam_target)

        # camera params
        camera_u = torch.arange(0, self.camera_props.width)
        camera_v = torch.arange(0, self.camera_props.height)
        self.camera_v2, self.camera_u2 = torch.meshgrid(camera_v, camera_u, indexing='ij')
        self.camera_u2 = to_torch(self.camera_u2, device=self.device)
        self.camera_v2 = to_torch(self.camera_v2, device=self.device)
        # point cloud params
        self.x_n_bar = self.cfg['env']['vision']['bar']['x_n']
        self.x_p_bar = self.cfg['env']['vision']['bar']['x_p']
        self.y_n_bar = self.cfg['env']['vision']['bar']['y_n']
        self.y_p_bar = self.cfg['env']['vision']['bar']['y_p']
        self.z_n_bar = self.cfg['env']['vision']['bar']['z_n']
        self.z_p_bar = self.cfg['env']['vision']['bar']['z_p']
        self.depth_bar = self.cfg['env']['vision']['bar']['depth']
        self.num_pc_downsample = self.cfg['env']['vision']['pointclouds']['numDownsample']
        self.num_pc_presample = self.cfg['env']['vision']['pointclouds']['numPresample']
        self.num_each_pt = self.cfg['env']['vision']['pointclouds']['numEachPoint']

        # init pytorch_renderer
        self.pytorch_renderer = PytorchBatchRenderer(num_view=6, img_size=self.image_size, center=self.table_center, device=self.device)
        # load pytorch_renderer view_matrix, convert to isaacgym axis
        self.pytorch_renderer_view_matrix = self.pytorch_renderer.camera_view_mat
        self.pytorch_renderer_view_matrix[:, :, [0, 2]] *= -1
        self.pytorch_renderer_view_matrix = self.pytorch_renderer_view_matrix[:, [2, 0, 1, 3], :]
        # load pytorch_renderer proj_matrix, convert to isaacgym axis
        self.pytorch_renderer_proj_matrix = self.pytorch_renderer.camera_proj_matrix
        self.pytorch_renderer_proj_matrix[:, [2, 3], :] *= -1
        # load pytorch_renderer proj_matrix
        self.pytorch_renderer_vinv_matrix = torch.inverse(self.pytorch_renderer_view_matrix)
        # repeat pytorch_renderer params with num_envs
        self.pytorch_renderer_view_matrix = self.pytorch_renderer_view_matrix.repeat(self.num_envs, 1, 1, 1)
        self.pytorch_renderer_proj_matrix = self.pytorch_renderer_proj_matrix.repeat(self.num_envs, 1, 1, 1)
        self.pytorch_renderer_vinv_matrix = self.pytorch_renderer_vinv_matrix.repeat(self.num_envs, 1, 1, 1)

        # get gym GPU state tensors
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        # create sensors and tensors for full_state obeservations
        if self.obs_type == "full_state" or self.asymmetric_obs:
            # create vec sensor
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)
            # create force sensor
            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs + self.num_object_dofs)
            self.dof_force_tensor = self.dof_force_tensor[:, :self.num_shadow_hand_dofs]
        # refresh tensor
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # shadow_hand_dof
        self.z_theta = torch.zeros(self.num_envs, device=self.device)
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]
        # rigid_body_states
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        # root_state_tensor
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()
        self.saved_root_tensor[self.object_indices, 9:10] = 0.0
        # control tensor
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs,-1)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        # utility tensor
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        # reset and success tensor
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.final_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.total_successes = 0
        self.total_resets = 0

        # use dynamic object visual features for ppo or dagger
        self.use_dynamic_visual_feats, self.object_visual_encoder_name = False, 'PN_128_scaled'
        self.object_points_visual_features = torch.zeros((self.num_envs, 128), device=self.device)
        if self.algo == 'ppo' and 'dynamic_object_visual_feature' in self.config['Modes'] and self.config['Modes']['dynamic_object_visual_feature']:
            self.use_dynamic_visual_feats = True
        if self.algo == 'dagger_value' and 'dynamic_object_visual_feature' in self.config['Distills'] and self.config['Distills']['dynamic_object_visual_feature']:
            self.use_dynamic_visual_feats = True
            self.object_visual_encoder_name = self.config['Distills']['object_visual_feature']
        # load and apply dynamic visual feature encoder
        if self.use_dynamic_visual_feats or self.config['Save']:
            # load object visual scaler
            self.object_visual_scaler = np.load(osp.join(BASE_DIR, 'dexgrasp/autoencoding/ckpts/{}/scaler.npy'.format(self.object_visual_encoder_name)), allow_pickle=True).item()
            self.object_visual_scaler_mean = torch.tensor(self.object_visual_scaler.mean_, device=self.device)
            self.object_visual_scaler_scale = torch.tensor(self.object_visual_scaler.scale_, device=self.device)
            # load object visual encoder
            self.object_visual_encoder = AutoencoderPN(k=int(self.object_visual_encoder_name.split('_')[1]), num_points=1024)
            self.object_visual_encoder.load_state_dict(torch.load(osp.join(BASE_DIR, 'dexgrasp/autoencoding/ckpts/{}/029900.pth'.format(self.object_visual_encoder_name))))
            self.object_visual_encoder.to(self.device)
            self.object_visual_encoder.eval()

    # create sim
    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        # create sim following BaseTask
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # create ground plane
        self._create_ground_plane()
        # create envs
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    # create ground
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    # create envs
    def _create_envs(self, num_envs, spacing, num_per_row):
        
        # # # ---------------------- Load Object_Scale_Dict ---------------------- # #
        if self.object_scale_file == "Default":
            # load object_scale_dict from shadow_hand_grasp.yaml
            self.object_scale_dict = self.cfg['env']['object_code_dict']
            self.object_scale_list = ['{}/{}'.format(obj_name, scale) for obj_name, scale_list in self.object_scale_dict.items() for scale in scale_list]
            self.object_code_list = list(self.object_scale_dict.keys())
        else:
            # load object_scale_dict from object_scale_file: train_set_results.yaml
            yaml_file = osp.join(BASE_DIR, 'results/configs/{}'.format(self.object_scale_file))
            # locate distill_group objects
            if self.group is not None:
                # test distill_group with single object
                if self.is_testing:
                    self.object_line_list, self.object_scale_list, self.object_scale_dict = \
                        load_object_scale_group_yaml(yaml_file, self.group, self.start_line)
                # train distill_group with group objects
                else:
                    self.object_line_list, self.object_scale_list, self.object_scale_dict = \
                        load_object_scale_group_yaml(yaml_file, self.group)
            # locate train_single object
            else:
                self.object_line_list, self.object_scale_list, self.object_scale_dict = \
                    load_object_scale_result_yaml(yaml_file, self.start_line, self.end_line, self.shuffle_dict)
            self.object_code_list = list(self.object_scale_dict.keys())

        # self.env_object_scale = [object_code/scale_str, ]
        # self.object_scale_dict = {object_code: [scale], }
        # self.object_scale_list = [object_code/scale, ]
        # self.object_code_list = [object_code, ]

        # self.visual_feat_data[object_code][scale_str]
        # self.object_asset_dict[object_code][scale_str]
        # self.grasp_data[object_code][scale_str]

        # # # ---------------------- Init Settings ---------------------- # #
        # locate Asset path
        self.assets_path = self.cfg["env"]['asset']['assetRoot']
        if not osp.exists(self.assets_path): self.assets_path = '../' + self.assets_path
        if not osp.exists(self.assets_path): self.assets_path = '/mnt/blob/Desktop/Assets'
        
        # init goal_conditional settings
        self.repose_z = self.cfg['env']['repose_z']
        self.goal_cond = self.cfg["env"]["goal_cond"]
        self.random_prior = self.cfg['env']['random_prior']
        self.random_time = self.cfg["env"]["random_time"]
        self.target_qpos = torch.zeros((self.num_envs, 22), device=self.device)
        self.target_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_init_euler_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.object_init_z = torch.zeros((self.num_envs, 1), device=self.device)
        # random_time: True
        self.random_time = True if self.config['Modes']['random_time'] else False


        # # # ---------------------- Load ShadowHand, Object Assets ---------------------- # #
        # valid scales
        self.scale2str = {0.06: '006', 0.08: '008', 0.10: '010', 0.12: '012', 0.15: '015'}
        self.str2scale = {'006': 0.06, '008': 0.08, '010': 0.10, '012': 0.12, '015': 0.15}

        # load shadow_hand asset
        shadow_hand_asset, shadow_hand_start_pose, shadow_hand_dof_props = self._load_shadow_hand_assets(self.assets_path)
        
        # load grasp pose data
        self._load_grasp_pose_data(self.assets_path, self.scale2str)
        # load object visual feature
        self._load_object_visual_feature(self.assets_path, self.scale2str)
        # load object, goal, table assets
        self.object_asset_dict, goal_asset, table_asset, object_start_pose, goal_start_pose, table_pose = \
            self._load_object_table_goal_assets(self.assets_path, self.scale2str)
        # create point asset for point cloud visualization
        point_cloud_asset = self._create_point_asset()

        # load table_texture
        table_texture_files = osp.join(self.assets_path, "textures/texture_stone_stone_texture_0.jpg")
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)
        
        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies * 1 + 2 * self.num_object_bodies + 1  ##
        max_agg_shapes = self.num_shadow_hand_shapes * 1 + 20 * self.num_object_shapes + 1  ##

        # init env info list
        self.envs, self.env_object_scale = [], []
        # init shadow hand info list
        self.shadow_handles, self.hand_indices, self.fingertip_indices = [], [], []
        # init handles for object, table, and goal
        self.object_handles, self.table_handles, self.goal_handles = [], [], []
        # init initial info list
        self.object_init_state, self.goal_init_state, self.hand_start_states = [], [], []
        self.object_scale_buf, self.object_init_mesh = {}, {'mesh': [], 'mesh_vertices': [], 'mesh_faces': [], 'points': [], 'points_centered': [], 'pca_axes': [], 'init_states_train': [], 'init_states_test': []}
        # init indices list
        self.object_indices, self.goal_object_indices, self.table_indices = [], [], []
        # init hand and object point list
        self.hand_point_handles, self.hand_point_indices, self.hand_point_nums = [], [], 40
        self.object_point_handles, self.object_point_indices = [], []
        # init env_origin
        self.env_origin = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # init env_object_scale_id
        self.env_object_scale_id = []

        # # ---------------------- Create Envs ---------------------- # #
        print('Create num_envs', self.num_envs)
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        # create envs
        loop = tqdm.tqdm(range(self.num_envs))
        for env_id in loop:
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            # aggregate_mode == 1
            if self.aggregate_mode >= 1: self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            loop.set_description('Creating env {}/{}'.format(env_id, self.num_envs))
            
            # # ---------------------- Create ShadowHand Actor ---------------------- # #
            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", env_id, -1, SEGMENT_ID['hand'][0])
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z,
                                           shadow_hand_start_pose.r.w, 0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # assign colors and textures for shadow_hand rigid body
            hand_color_list = [[147/255, 215/255, 160/255], [255/255, 0/255, 0/255], [255/255, 128/255, 0/255], 
                               [255/255, 255/255, 0/255], [0/255, 255/255, 0/255], [0/255, 255/255, 255/255], [0/255, 0/255, 255/255]]
            # [palm, thumb, index, middle, ring, little], [1, 23, 5, 9, 13, 18]
            hand_rigid_body_index = [[0,1,2], [3,4,5,6], [7,8,9,10], [11,12,13], [14,15,16,17,18,19], [20,21,22,23,24,25]]
            for n in self.agent_index[0]:
                for m in n:
                    for o in hand_rigid_body_index[m]:
                        hand_color = hand_color_list[m]
                        hand_color = [128/255, 215/255, 128/255]  # unique hand color
                        self.gym.set_rigid_body_color(env_ptr, shadow_hand_actor, o, gymapi.MESH_VISUAL, gymapi.Vec3(*hand_color))

            # create fingertip force-torque sensors
            if self.obs_type == "full_state" or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)


            # # ---------------------- Create Object/Table/Goal Actor ---------------------- # #
            # randomly or uniformly pick env object_scale from object_scale_list
            if self.shuffle_env: temp_id = random.randint(0, len(self.object_scale_list)-1)
            else:
                if self.num_envs % len(self.object_scale_list) == 0: temp_id = env_id // (self.num_envs // len(self.object_scale_list))
                else: temp_id = int(env_id / self.num_envs * len(self.object_scale_list))
            # append env_object_scale_id
            self.env_object_scale_id.append(temp_id)
            # choose object_scale from object_scale_list
            object_scale_name = self.object_scale_list[temp_id]
            # unpack object_code and scale
            object_code = '{}/{}'.format(object_scale_name.split('/')[0],  object_scale_name.split('/')[1])
            scale = float(object_scale_name.split('/')[2])
            scale_str = self.scale2str[scale]
            
            # update env object buff
            self.object_id_buf[env_id] = env_id
            self.object_scale_buf[env_id] = scale
            self.visual_feat_buf[env_id] = self.visual_feat_data[object_code][scale_str]

            # load object_asset_info
            if self.use_object_asset_dict:
                object_asset_info = self.object_asset_dict[object_code][scale_str]
            else:
                object_asset_info = self._load_object_asset_info(self.assets_path, object_code, scale_str)
            
            # # print_cpu_gpu_usage
            # print_cpu_gpu_usage(int(self.device.split(':')[-1]))
            # print('env_object_scale:', object_asset_info[1])
            self.env_object_scale.append(object_asset_info[1])
            # create object actor
            object_handle = self.gym.create_actor(env_ptr, object_asset_info[0], object_start_pose, "object", env_id, 0, SEGMENT_ID['object'][0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.gym.set_actor_scale(env_ptr, object_handle, 1.0)
            # record object init state
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                           object_start_pose.r.w, 0, 0, 0, 0, 0, 0])
            # record object init mesh and points
            self.object_init_mesh['mesh'].append(object_asset_info[2])
            self.object_init_mesh['mesh_vertices'].append(object_asset_info[2].vertices)
            self.object_init_mesh['mesh_faces'].append(object_asset_info[2].faces)
            self.object_init_mesh['points'].append(object_asset_info[3])
            self.object_init_mesh['points_centered'].append(object_asset_info[4])
            self.object_init_mesh['pca_axes'].append(object_asset_info[5])
            self.object_init_mesh['init_states_train'].append(object_asset_info[6])
            self.object_init_mesh['init_states_test'].append(object_asset_info[7])

            # create goal actor
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", env_id + self.num_envs, 0, SEGMENT_ID['goal'][0])
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)
            self.gym.set_actor_scale(env_ptr, goal_handle, 1.0)
            # record goal init state
            self.goal_init_state.append([goal_start_pose.p.x, goal_start_pose.p.y, goal_start_pose.p.z,
                                         goal_start_pose.r.x, goal_start_pose.r.y, goal_start_pose.r.z,
                                         goal_start_pose.r.w, 0, 0, 0, 0, 0, 0])
            
            # create table actor
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", env_id, -1, SEGMENT_ID['table'][0])
            self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            # create point cloud actor for visualization
            self.hand_point_handles.append([])
            self.object_point_handles.append([])
            if self.render_point_clouds:
                # create hand point cloud actors
                for npoint in range(self.hand_point_nums):
                    # create point_handle
                    pose = gymapi.Transform()
                    point_handle = self.gym.create_actor(env_ptr, goal_asset, pose, "point_cloud", env_id + self.num_envs, 0, SEGMENT_ID['other'][0])
                    color_value = (255 / self.hand_point_nums) * npoint
                    color = [color_value/255, color_value/255, color_value/255]
                    self.gym.set_rigid_body_color(env_ptr, point_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color))
                    # append point_handle and point_indices
                    self.hand_point_handles[-1].append(point_handle)
                    self.hand_point_indices.append(self.gym.get_actor_index(env_ptr, point_handle, gymapi.DOMAIN_SIM))
                # create object point cloud actors 
                for point in object_asset_info[3]:
                    # create point_handle
                    pose = gymapi.Transform()
                    pose.p = gymapi.Vec3(*point)
                    point_handle = self.gym.create_actor(env_ptr, point_cloud_asset, pose, "point_cloud", env_id + self.num_envs, 0, SEGMENT_ID['other'][0])
                    color = [255/255, 125/255, 0/255]
                    self.gym.set_rigid_body_color(env_ptr, point_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color))
                    # append point_handle and point_indices
                    self.object_point_handles[-1].append(point_handle)
                    self.object_point_indices.append(self.gym.get_actor_index(env_ptr, point_handle, gymapi.DOMAIN_SIM))

            # set table and object friction
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            table_shape_props[0].friction = 1
            object_shape_props[0].friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)
            # set table, object, goal color
            self.gym.set_rigid_body_color(env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*[90/255, 90/255, 173/255]))
            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*[150/255, 150/255, 150/255]))
            self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*[173/255, 90/255, 90/255]))
            
            # create env cameras, update camera_handle_list, depth_tensors...
            if self.num_cameras > 0: self.create_env_cameras(env_ptr, env_id, self.camera_props, self.camera_eye_list, self.camera_lookat_list, self.render_env_list)

            # stop aggregate actors
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
            # append env_ptr
            self.envs.append(env_ptr)

        # init camera params
        if self.num_cameras > 0:
            self.view_mat = torch.stack([torch.stack(i) for i in self.camera_view_mat_list])
            self.vinv_mat = torch.stack([torch.stack(i) for i in self.camera_vinv_mat_list])
            self.proj_matrix = torch.stack([torch.stack(i) for i in self.camera_proj_mat_list])
        
        # send items to torch
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_init_state = to_torch(self.goal_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.goal_init_state.clone()
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)
    
        # send object init points (nenv, 1024, 3) and axes (nenv, 3, 3) to tensor
        self.object_init_mesh['points'] = to_torch(np.stack(self.object_init_mesh['points'], axis=0), device=self.device, dtype=torch.float)
        self.object_init_mesh['points_centered'] = to_torch(np.stack(self.object_init_mesh['points_centered'], axis=0), device=self.device, dtype=torch.float)
        self.object_init_mesh['pca_axes'] = to_torch(np.stack(self.object_init_mesh['pca_axes'], axis=0), device=self.device, dtype=torch.float)
        # send init states to tensor
        if 'static_init' in self.config['Modes'] and self.config['Modes']['static_init']:
            self.object_init_mesh['init_states_train'] = to_torch(np.stack(self.object_init_mesh['init_states_train'], axis=0), device=self.device, dtype=torch.float)
            self.object_init_mesh['init_states_test'] = to_torch(np.stack(self.object_init_mesh['init_states_test'], axis=0), device=self.device, dtype=torch.float)
        # append hand and object point indices
        self.hand_point_indices = to_torch(self.hand_point_indices, dtype=torch.long, device=self.device)
        self.object_point_indices = to_torch(self.object_point_indices, dtype=torch.long, device=self.device)

        # check load single or multiple objects
        self.load_single_object = self.env_object_scale_id.count(self.env_object_scale_id[0]) == len(self.env_object_scale_id)
        # stack single object mesh
        if self.load_single_object:
            self.object_init_mesh['mesh_vertices'] = to_torch(np.stack(self.object_init_mesh['mesh_vertices'], axis=0), device=self.device, dtype=torch.float)
            self.object_init_mesh['mesh_faces'] = to_torch(np.stack(self.object_init_mesh['mesh_faces'], axis=0), device=self.device, dtype=torch.long)

    # load grasp poses for object_scales
    def _load_grasp_pose_data(self, assets_path, scale2str):
        # load grasp_data from assets/datasetv4.1_posedata.npy
        self.grasp_data_np = np.load(osp.join(assets_path, 'datasetv4.1_posedata.npy'), allow_pickle=True).item()
        # init grasp_data
        keys_to_convert = ['target_qpos', 'target_hand_pos', 'target_hand_rot', 'object_euler_xy', 'object_init_z']
        self.grasp_data = {object_code: {scale2str[scale]: {key: None for key in keys_to_convert} for scale in self.grasp_data_np[object_code]} for object_code in self.grasp_data_np}
        # update grasp_data: data_per_object for target_pose, object_pose
        for object_code in list(self.grasp_data_np.keys()):
            if object_code not in self.object_code_list: continue
            data_per_object = self.grasp_data[object_code]
            data_per_object_np = self.grasp_data_np[object_code]
            for scale in list(data_per_object_np.keys()):
                if scale not in self.object_scale_dict[object_code]: continue
                for key in keys_to_convert:
                    data_per_object[scale2str[scale]][key] = [torch.tensor(item, dtype=torch.float, device=self.device) for item in data_per_object_np[scale][key]]
                    # In UniDexGrasp++ we don't use the grasp pose in data so we simply set this to 0
                    if key in ['target_qpos', 'target_hand_pos', 'target_hand_rot']:
                        data_per_object[scale2str[scale]][key] = [value * 0 for value in data_per_object[scale2str[scale]][key]]
    
    # load shadow_hand_assets and poses
    def _load_shadow_hand_assets(self, assets_path):        
        # locate shadow_hand Assets 
        shadow_hand_asset_file = "mjcf/open_ai_assets/hand/shadow_hand.xml"
        if "asset" in self.cfg["env"]:
            shadow_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file)

        # TODO: load shadow hand model
        self.shadow_hand_model = ShadowHandModel('./hand_assets/shadow_hand_render.xml', './hand_assets/open_ai_assets/stls/hand', simplify_mesh=True, device=self.device)

        # set shadow_hand AssetOptions
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        # load shadow_hand_asset
        shadow_hand_asset = self.gym.load_asset(self.sim, assets_path, shadow_hand_asset_file, asset_options)
        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)  # 24
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)  # 20
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)  # 22
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)  # 18
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)  # 4

        # # load shadow_hand_body_names
        # self.num_shadow_hand_body_names = self.gym.get_asset_rigid_body_names(shadow_hand_asset)

        # valid rigid body indices
        self.valid_shadow_hand_bodies = [1, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 23]

        # tendon set up
        limit_stiffness, t_damping = 30, 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)
        # set tendon limit_stiffness and damping
        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)

        # locate shadow_hand actuated_dof
        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_actuators)]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in actuated_dof_names]

        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)
        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()
        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)
        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        # static init object states
        if 'static_init' in self.config['Modes'] and self.config['Modes']['static_init']:
            hand_shift = 0.08
            # init shadow_hand pose: top of table 0.2, face down, center palm
            shadow_hand_start_pose = gymapi.Transform()
            shadow_hand_start_pose.p = gymapi.Vec3(0.0, hand_shift, self.table_dims.z + 0.2)  # gymapi.Vec3(0.1, 0.1, 0.65)
            shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, 0)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)
        else:
            # init shadow_hand pose: top of table 0.2, face down
            shadow_hand_start_pose = gymapi.Transform()
            shadow_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, self.table_dims.z + 0.2)  # gymapi.Vec3(0.1, 0.1, 0.65)
            shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, 0)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)

        # locate hand body index   
        body_names = {'wrist': 'robot0:wrist', 'palm': 'robot0:palm', 'thumb': 'robot0:thdistal',
                      'index': 'robot0:ffdistal', 'middle': 'robot0:mfdistal', 'ring': 'robot0:rfdistal', 'little': 'robot0:lfdistal'}
        self.hand_body_idx_dict = {}
        for name, body_name in body_names.items():
            self.hand_body_idx_dict[name] = self.gym.find_asset_rigid_body_index(shadow_hand_asset, body_name)
        # locate fingertip_handles indices [5, 9, 13, 18, 23]
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]

        # create fingertip force sensors, if needed
        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)
        
        return shadow_hand_asset, shadow_hand_start_pose, shadow_hand_dof_props
    
    # load object visual feature: self.visual_feat_data = {object_code: {scale: feat, }, }
    def _load_object_visual_feature(self, assets_path, scale2str):
        # locate visual_feature
        visual_feat_root = osp.realpath(osp.join(assets_path, 'meshdatav3_pc_feat'))
        # loop over all object_code_list, load visual_feature
        self.visual_feat_data, self.visual_feat_buf = {}, torch.zeros((self.num_envs, 64), device=self.device)
        loop = tqdm.tqdm(range(len(self.object_code_list)))
        for object_id in loop:
            object_code = self.object_code_list[object_id]
            loop.set_description('Loading visual_feature {}'.format(object_id))
            # load object visual feature for all object_scale
            self.visual_feat_data[object_code] = {}
            for scale in self.object_scale_dict[object_code]:
                scale_str = scale2str[scale]
                file_dir = osp.join(visual_feat_root, f'{object_code}/pc_feat_{scale_str}.npy')
                with open(file_dir, 'rb') as f: feat = np.load(f)
                self.visual_feat_data[object_code][scale_str] = torch.tensor(feat, device=self.device)

    # load object asset info: [asset, object_code/scale_str, mesh, points, pca_axes]
    def _load_object_asset_info(self, assets_path, object_code, scale_str):
        # locate mesh folder
        mesh_path = osp.join(assets_path, 'meshdatav3_scaled')
        # load object asset
        scaled_object_asset_file = object_code + f"/coacd/coacd_{scale_str}.urdf"
        scaled_object_asset = self.gym.load_asset(self.sim, mesh_path, scaled_object_asset_file, self.object_asset_options)
        # load object mesh and points
        scaled_object_mesh_file = os.path.join(mesh_path, object_code + f"/coacd/decomposed_{scale_str}.obj")
        scaled_object_mesh = trimesh.load(scaled_object_mesh_file)
        scaled_object_points, _ = trimesh.sample.sample_surface(scaled_object_mesh, 1024)
        # apply PCA to find the axis
        pca = PCA(n_components=3)
        pca.fit(scaled_object_points)
        pca_axes = pca.components_
        # locate and load object pc_fps
        scaled_object_pc_file = osp.join(assets_path, 'meshdatav3_pc_fps', object_code + f"/coacd/pc_fps1024_{scale_str}.npy")
        with open(scaled_object_pc_file, 'rb') as f: scaled_object_pc_fps = np.asarray(np.load(f))[:, :3]
        # locate and load static inital object states
        if 'static_init' in self.config['Modes'] and self.config['Modes']['static_init']:
            object_init_states = load_pickle(osp.join(assets_path, 'meshdatav3_init', object_code + f"/object_init_{scale_str}.pkl" ))
        else: object_init_states = {'train': None, 'test': None}
        # TODO: simplify object mesh for rendering
        scaled_object_mesh = simplify_trimesh(scaled_object_mesh, ratio=0.1, min_faces=500)
        # return object_asset_info
        return [scaled_object_asset, '{}/{}'.format(object_code, scale_str), scaled_object_mesh, scaled_object_pc_fps, scaled_object_pc_fps-scaled_object_pc_fps.mean(0), pca_axes, object_init_states['train'], object_init_states['test']]

    # load object, table and goal assets: object_asset_dict = {object_code: {scale: [asset, object_name/scale_name, ...], }, }
    def _load_object_table_goal_assets(self, assets_path, scale2str):
        # init object_asset_dict
        object_asset_dict = {}
        self.use_object_asset_dict = True if len(self.object_scale_list) <= 500 else False
        # create object_asset_options
        self.object_asset_options = gymapi.AssetOptions()
        self.object_asset_options.density = 500
        self.object_asset_options.fix_base_link = False
        self.object_asset_options.use_mesh_materials = True
        self.object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        self.object_asset_options.override_com = True
        self.object_asset_options.override_inertia = True
        self.object_asset_options.vhacd_enabled = True
        self.object_asset_options.vhacd_params = gymapi.VhacdParams()
        self.object_asset_options.vhacd_params.resolution = 300000
        self.object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        # loop over all object_code_list, load object data
        object_asset, self.num_object_bodies, self.num_object_shapes = None, -1, -1
        loop = tqdm.tqdm(range(len(self.object_code_list)))
        for object_id in loop:
            object_code = self.object_code_list[object_id]
            loop.set_description('Loading object_mesh {}'.format(object_id))
            if not self.use_object_asset_dict and object_id != 0: continue
            # load object_asset for all object_scale
            object_asset_dict[object_code] = {}
            for scale in self.object_scale_dict[object_code]:
                scale_str = scale2str[scale]
                object_asset_info = self._load_object_asset_info(assets_path, object_code, scale_str)
                object_asset_dict[object_code][scale_str] = object_asset_info
                # update num_object_bodies, num_object_shapes
                self.num_object_bodies = max(self.num_object_bodies, self.gym.get_asset_rigid_body_count(object_asset_info[0]))
                self.num_object_shapes = max(self.num_object_shapes, self.gym.get_asset_rigid_shape_count(object_asset_info[0]))
                # update object_asset
                if object_asset is None: object_asset = object_asset_info[0]

        # get object dof properties: lower and upper limits
        self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
        self.object_dof_props = self.gym.get_asset_dof_properties(object_asset)
        self.object_dof_lower_limits = to_torch([self.object_dof_props['lower'][i] for i in range(self.num_object_dofs)], device=self.device)
        self.object_dof_upper_limits = to_torch([self.object_dof_props['upper'][i] for i in range(self.num_object_dofs)], device=self.device)

        # create goal_asset_options
        goal_asset_options = gymapi.AssetOptions()
        goal_asset_options.density = 500
        goal_asset_options.fix_base_link = False
        goal_asset_options.disable_gravity = True
        goal_asset_options.use_mesh_materials = True
        goal_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        goal_asset_options.override_com = True
        goal_asset_options.override_inertia = True
        goal_asset_options.vhacd_enabled = True
        goal_asset_options.vhacd_params = gymapi.VhacdParams()
        goal_asset_options.vhacd_params.resolution = 300000
        goal_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # create goal_asset
        goal_asset = self.gym.create_sphere(self.sim, 0.01, goal_asset_options)

        # create table_asset
        table_asset_options = gymapi.AssetOptions()
        # if 'static_init' in self.config['Modes'] and self.config['Modes']['static_init']: table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, table_asset_options)

        # init object pose: top of table 0.1
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, self.table_dims.z + 0.1)  # gymapi.Vec3(0.0, 0.0, 0.72)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        # init goal pose: top of table 0.3
        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.2)
        self.goal_displacement_tensor = to_torch([self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        goal_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        goal_start_pose.p.z -= 0.0
        # init table pose: top of table
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * self.table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        # create table_mesh
        self.table_mesh = trimesh.creation.box(extents=(self.table_dims.x, self.table_dims.y, self.table_dims.z))
        self.table_mesh.vertices += np.array([0.0, 0.0, 0.5 * self.table_dims.z])
        # repeat table mesh with Nenvs
        self.table_vertices = torch.tensor(self.table_mesh.vertices, dtype=torch.float).repeat(self.num_envs, 1, 1).to(self.device)
        self.table_faces = torch.tensor(self.table_mesh.faces, dtype=torch.long).repeat(self.num_envs, 1, 1).to(self.device)
        self.table_colors = torch.tensor(SEGMENT_ID['table'][1]).repeat(self.table_vertices.shape[0], self.table_vertices.shape[1], 1).to(self.device) / 255.
        # self.table_labels = torch.tensor(SEGMENT_ID['table'][0]).repeat(self.table_vertices.shape[0], self.table_vertices.shape[1], 1).to(self.device)

        return object_asset_dict, goal_asset, table_asset, object_start_pose, goal_start_pose, table_pose

    # create single point asset
    def _create_point_asset(self):
        # Create the point cloud actors in the environment
        sphere_asset_options = gymapi.AssetOptions()
        sphere_asset_options.density = 500
        sphere_asset_options.fix_base_link = True
        sphere_asset_options.disable_gravity = True
        sphere_asset_options.armature = 0.0
        sphere_asset_options.thickness = 0.01
        sphere_asset = self.gym.create_sphere(self.sim, 0.002, sphere_asset_options)
        return sphere_asset

    # # ---------------------- Create Env Cameras ---------------------- # #
    # create camera configs
    def create_cfg_cameras(self):
        # create camera properties
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = self.image_size
        self.camera_props.height = self.image_size
        self.camera_props.enable_tensors = True

        # create camera poses
        self.camera_eye_list = []
        self.camera_lookat_list = []
        camera_eye_list = CAMERA_PARAMS['eye']
        camera_lookat_list = CAMERA_PARAMS['lookat']
        for i in range(self.num_cameras):
            camera_eye = np.array(camera_eye_list[i]) + self.table_center
            camera_lookat = np.array(camera_lookat_list[i]) + self.table_center
            self.camera_eye_list.append(gymapi.Vec3(*list(camera_eye)))
            self.camera_lookat_list.append(gymapi.Vec3(*list(camera_lookat)))
        return
    
    # create camera handles and tensors for each view
    def create_env_cameras(self, env, env_id, camera_props, camera_eye_list, camera_lookat_list, render_env_list):
        # skip envs
        if render_env_list is not None and env_id not in render_env_list: return
        
        # init env camera_handles
        camera_handles = []
        # init depth, rgb and seg tensors
        depth_tensors, rgb_tensors, seg_tensors, view_mats, vinv_mats, proj_mats = [], [], [], [], [], []
        # locate env center
        origin = self.gym.get_env_origin(env)
        self.env_origin[env_id][0] = origin.x
        self.env_origin[env_id][1] = origin.y
        self.env_origin[env_id][2] = origin.z
        # init cameras
        for i in range(self.num_cameras):
            # create camera sensor
            camera_handle = self.gym.create_camera_sensor(env, camera_props)
            # load camera params
            camera_eye = camera_eye_list[i]
            camera_lookat = camera_lookat_list[i]
            self.gym.set_camera_location(camera_handle, env, camera_eye, camera_lookat)

            # append camera_handles
            camera_handles.append(camera_handle)

            # render depth tensor
            raw_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, camera_handle, gymapi.IMAGE_DEPTH)
            depth_tensor = gymtorch.wrap_tensor(raw_depth_tensor)
            depth_tensors.append(depth_tensor)
            # render rgb tensor
            raw_rgb_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, camera_handle, gymapi.IMAGE_COLOR)
            rgb_tensor = gymtorch.wrap_tensor(raw_rgb_tensor)
            rgb_tensors.append(rgb_tensor)
            # render seg tensor
            raw_seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, camera_handle, gymapi.IMAGE_SEGMENTATION)
            seg_tensor = gymtorch.wrap_tensor(raw_seg_tensor)
            seg_tensors.append(seg_tensor)
            # get camera view matrix
            view_mat = to_torch(self.gym.get_camera_view_matrix(self.sim, env, camera_handle), device=self.device)
            view_mats.append(view_mat)
            # get camera inverse view matrix
            vinv_mat = torch.inverse(view_mat)
            vinv_mats.append(vinv_mat)
            # get camera projection matrix
            proj_mat = to_torch(self.gym.get_camera_proj_matrix(self.sim, env, camera_handle), device=self.device)
            proj_mats.append(proj_mat)

        # append camera handle
        self.camera_handle_list.append(camera_handles)
        # append camera tensors
        self.camera_depth_tensor_list.append(depth_tensors)
        self.camera_rgb_tensor_list.append(rgb_tensors)
        self.camera_seg_tensor_list.append(seg_tensors)
        self.camera_view_mat_list.append(view_mats)
        self.camera_vinv_mat_list.append(vinv_mats)
        self.camera_proj_mat_list.append(proj_mats)
        # # save camera params
        # save_pickle(os.path.join(BASE_DIR, 'camera_params.pkl'), {'view_mat': np.transpose(self.view_mat[0].cpu().numpy(), (0, 2, 1)), 'proj_mat': np.transpose(self.proj_matrix[0].cpu().numpy(), (0, 2, 1))})
        return
    
    # TODO: render pytorch images and point_clouds from hand_object_states (Nenv, [3, 4, 22, 3, 4])
    def render_pytorch_images_points(self, hand_object_states, render_images=False, sample_points=False):
        # return without rendering
        if not render_images: return None, None

        # unpack hand_object_states: [3, 4, 22, 3, 4]
        hand_pos, hand_rot, hand_pose = hand_object_states[:, :3], hand_object_states[:, 3:3+4], hand_object_states[:, 3+4:3+4+22]
        object_pos, object_rot = hand_object_states[:, 3+4+22:3+4+22+3], hand_object_states[:, 3+4+22+3:]
        
        # get current shadow_hand vertices and faces
        self.shadow_hand_vertices, self.shadow_hand_faces, _ = self.shadow_hand_model.get_current_meshes(hand_pos, hand_rot, hand_pose)
        self.shadow_hand_colors = torch.tensor(SEGMENT_ID['hand'][1]).repeat(self.shadow_hand_vertices.shape[0], self.shadow_hand_vertices.shape[1], 1).to(self.device) / 255.
        # self.shadow_hand_labels = torch.tensor(SEGMENT_ID['hand'][0]).repeat(self.shadow_hand_vertices.shape[0], self.shadow_hand_vertices.shape[1], 1).to(self.device)

        # get current object vertices and faces
        self.object_vertices = batch_quat_apply(object_rot, self.object_init_mesh['mesh_vertices']) + object_pos.unsqueeze(1)
        self.object_faces = self.object_init_mesh['mesh_faces']
        self.object_colors = torch.tensor(SEGMENT_ID['object'][1]).repeat(self.object_vertices.shape[0], self.object_vertices.shape[1], 1).to(self.device) / 255.
        # self.object_labels = torch.tensor(SEGMENT_ID['object'][0]).repeat(self.object_vertices.shape[0], self.object_vertices.shape[1], 1).to(self.device)

        # combine shadow_hand and object meshes
        self.rendered_mesh_vertices = torch.cat([self.shadow_hand_vertices, self.object_vertices, self.table_vertices], dim=1)
        self.rendered_mesh_faces = torch.cat([self.shadow_hand_faces, self.object_faces+self.shadow_hand_vertices.shape[1], self.table_faces+self.shadow_hand_vertices.shape[1]+self.object_vertices.shape[1]], dim=1)
        self.rendered_mesh_colors = torch.cat([self.shadow_hand_colors, self.object_colors, self.table_colors], dim=1)
        # self.rendered_mesh_labels = torch.cat([self.shadow_hand_labels, self.object_labels, self.table_labels], dim=1)
        if self.repose_z: self.rendered_mesh_vertices[..., :3] = self.unpose_pc(self.rendered_mesh_vertices[..., :3])

        # render images (Nenv, Nview, H, W, RGBMD)
        rendered_images = self.pytorch_renderer.render_mesh_images(self.rendered_mesh_vertices[:, :, [1, 2, 0]], self.rendered_mesh_faces, self.rendered_mesh_colors)
        # rendered labels (Nenv, Nview, H, W)
        segmentation_labels = torch.stack([torch.tensor(SEGMENT_ID[label][1]) for label in SEGMENT_ID_LIST]).to(self.device) / 255.
        rendered_labels = torch.argmin(torch.norm(rendered_images[..., :3].unsqueeze(-2).repeat(1, 1, 1, 1, segmentation_labels.shape[0], 1) - segmentation_labels.reshape(1, 1, 1, 1, segmentation_labels.shape[0], segmentation_labels.shape[1]), dim=-1), dim=-1)
        # get final rendered_images (Nenv, Nview, H, W, RGBMDS)
        rendered_images = torch.cat([rendered_images, rendered_labels.unsqueeze(-1)], dim=-1)

        # return without sampling 
        if not sample_points: return rendered_images, None
        # render point_clouds (Nenv, Npoint, XYZS)
        if self.image_size==1024: self.num_pc_downsample = 4096
        rendered_points, others = self.render_camera_point_clouds(rendered_images[..., -2], rendered_images[..., -1], # self.vinv_mat, self.proj_matrix)
                                                                  self.pytorch_renderer_vinv_matrix, self.pytorch_renderer_proj_matrix, render_scene_only=True)
        return rendered_images, rendered_points

    # render camera images and point_clouds
    def render_camera_images_points(self, render_images=False, sample_points=False):        
        # get depth(n_env, n_cam, h, w), seg(n_env, n_cam, h, w) tensors
        depth_tensor = torch.stack([torch.stack(i) for i in self.camera_depth_tensor_list])
        rgb_tensor = torch.stack([torch.stack(i) for i in self.camera_rgb_tensor_list])
        seg_tensor = torch.stack([torch.stack(i) for i in self.camera_seg_tensor_list])
        # view_mat = torch.stack([torch.stack(i) for i in self.camera_view_mat_list])
        # vinv_mat = torch.stack([torch.stack(i) for i in self.camera_vinv_mat_list])
        # proj_matrix = torch.stack([torch.stack(i) for i in self.camera_proj_mat_list])

        # render each camera view, sample point clouds
        env_camera_images, points_fps, others = None, None, None
        if sample_points:
            points_fps, others = self.render_camera_point_clouds(depth_tensor, seg_tensor, self.vinv_mat, self.proj_matrix, render_scene_only=True)
            if not render_images: return env_camera_images, points_fps, others

        # init env_camera_images
        env_camera_images = {'depth': [], 'rgb': [], 'seg': []}
        # append env images
        for env_id in range(depth_tensor.shape[0]):
            # append env images
            env_camera_images['depth'].append(grid_camera_images(depth_tensor[env_id], border=False).float())
            env_camera_images['rgb'].append(grid_camera_images(rgb_tensor[env_id], border=False).float())
            env_camera_images['seg'].append(grid_camera_images(seg_tensor[env_id], border=False))

        # grid env images
        for key in ['depth', 'rgb', 'seg']:
            env_camera_images[key] = grid_camera_images(env_camera_images[key], [int(self.num_envs ** 0.5), int(self.num_envs ** 0.5)])

        # convert depth value to 0, 255
        env_camera_images['depth'] = 255 * torch.where(-env_camera_images['depth'] > 1., 1., -env_camera_images['depth'])
        # set rgb background as white
        env_camera_images['rgb'][env_camera_images['seg'] == SEGMENT_ID['back'][0]] = 255
        # set seg color
        temp = torch.zeros_like(env_camera_images['seg']).unsqueeze(2).expand(-1, -1, 3).float()
        for k, v in SEGMENT_ID.items():
            indices = torch.where(env_camera_images['seg'] == v[0])
            temp[indices[0], indices[1]] = torch.tensor(v[1]).float().to(self.device)
        env_camera_images['seg'] = temp
        return env_camera_images, points_fps, others

    # render scene, hand, object point clouds
    def render_camera_point_clouds(self, depth_tensor, seg_tensor, vinv_mat, proj_matrix, render_scene_only=True):
        # init point and valid list
        point_list, valid_list = [], []
        # get pixel point from depth, rgb, and seg images
        for i in range(1, depth_tensor.shape[1]):
            # (num_envs, num_pts, 4) (num_envs, num_pts)
            point, valid = depth_image_to_point_cloud_GPU_batch(depth_tensor[:, i], seg_tensor[:, i],
                                                                vinv_mat[:, i], proj_matrix[:, i], self.camera_u2, self.camera_v2, 
                                                                self.camera_props.width, self.camera_props.height, self.depth_bar, self.device,
                                                                # self.z_p_bar, self.z_n_bar
                                                                )
            point_list.append(point)
            valid_list.append(valid)

        # shift points (num_envs, 256*256 * num_cameras, 4)
        points = torch.cat(point_list, dim=1)
        points[:, :, :3] -= self.env_origin.view(self.num_envs, 1, 3)
        # get final valid mask
        depth_mask = torch.cat(valid_list, dim=1)
        x_mask = (points[:, :, 0] > self.x_n_bar) * (points[:, :, 0] < self.x_p_bar)
        y_mask = (points[:, :, 1] > self.y_n_bar) * (points[:, :, 1] < self.y_p_bar)
        z_mask = (points[:, :, 2] > self.z_n_bar) * (points[:, :, 2] < self.z_p_bar)
        s_mask = ((points[:, :, -1] == SEGMENT_ID['hand'][0]) + (points[:, :, -1] == SEGMENT_ID['object'][0])) > 0
        valid = depth_mask * x_mask * y_mask * z_mask * s_mask

        # get valid point_nums for each env (num_envs,)
        now, point_nums, points_list = 0, valid.sum(dim=1), []
        # (num_envs, num_valid_pts_total, 4)
        valid_points = points[valid]
        
        # presample, make num_pts equal for each env
        for env_id, point_num in enumerate(point_nums):
            if point_num == 0:
                points_list.append(torch.zeros(self.num_pc_presample, valid_points.shape[-1]).to(self.device))
            else:
                # print('env{}_____point_num = {}_____'.format(env_id, point_num))
                points_all = valid_points[now : now + point_num]
                random_ids = torch.randint(0, points_all.shape[0], (self.num_pc_presample,), device=self.device, dtype=torch.long)
                points_all_rnd = points_all[random_ids]
                points_list.append(points_all_rnd)
                now += point_num
        
        assert len(points_list) == self.num_envs, f'{self.num_envs - len(points_list)} envs have 0 point'
        # (num_envs, num_pc_presample)
        points_batch = torch.stack(points_list)

        # clean points
        def clean_points(points):
            if torch.sum(points[..., -1] == 0) == 0: return points
            # locate target points
            indices = torch.nonzero(points[..., -1] == 0)
            # change target points
            for n in range(indices.shape[0]):
                if torch.sum(points[indices[n][0], :, -1] != 0) == 0: continue
                points[indices[n][0]][indices[n][1]] = points[indices[n][0]][points[indices[n][0], :, -1] != 0][0]
            return points
        
        # render scene points
        points_fps, _ = sample_farthest_points(points_batch, K=self.num_pc_downsample*2 if render_scene_only else self.num_pc_downsample)
        # render hand and object points
        if not render_scene_only:
            # sample points with target sample_num
            num_sample_dict = self.cfg['env']['vision']['pointclouds']['numSample']
            zeros = torch.zeros((self.num_envs, self.num_pc_presample), device=self.device).to(torch.long)
            idx = torch.arange(self.num_envs * self.num_pc_presample, device=self.device).view(self.num_envs, self.num_pc_presample).to(torch.long)
            # mask first point
            points_batch[0, 0, :] *= 0.
            # extract hand, object points
            hand_idx = torch.where(points_batch[:, :, -1] == SEGMENT_ID['hand'][0], idx, zeros)
            hand_pc = points_batch.view(-1, points_batch.shape[-1])[hand_idx]
            object_idx = torch.where(points_batch[:, :, -1] == SEGMENT_ID['object'][0], idx, zeros)
            object_pc = points_batch.view(-1, points_batch.shape[-1])[object_idx]
            # sample hand, object points
            hand_fps, _ = sample_farthest_points(hand_pc, K=self.num_pc_downsample)
            object_fps, _ = sample_farthest_points(object_pc, K=self.num_pc_downsample)
            # clean hand, object points
            hand_fps = clean_points(hand_fps)
            object_fps = clean_points(object_fps)
            # concat hand, object points
            points_fps = torch.cat([points_fps, hand_fps, object_fps], dim=1)

        # repose points_fps
        if self.repose_z: points_fps[..., :3] = self.unpose_pc(points_fps[..., :3])

        # others
        others = {}
        return points_fps, others


    # # ---------------------- Physics Simulation Steps ---------------------- # #

    # pre_physics_step, reset envs, apply actions for ShadowHand base and joints
    def pre_physics_step(self, actions):
        # generate object initial states
        if self.config['Init']: actions *= 0
        # get env_ids to reset
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        # reset envs
        if len(env_ids) > 0:
            # zero actions before reset
            if 'reset_actions' in self.config['Modes'] and self.config['Modes']['reset_actions']: actions[env_ids] *= 0.
            if 'static_init' in self.config['Modes'] and self.config['Modes']['static_init']: actions[env_ids] *= 0.
            self.reset(env_ids, goal_env_ids)

        # apply control actions
        self.get_pose_quat()
        actions[:, 0:3] = self.pose_vec(actions[:, 0:3])
        actions[:, 3:6] = self.pose_vec(actions[:, 3:6])
        self.actions = actions.clone().to(self.device)

        # compute and apply forces and torques for ShadowHand Base
        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            # set 18 hand joint position targets
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 6:],self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:,self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            # apply hand base force and torque
            self.apply_forces[:, 1, :] = self.actions[:, 0:3] * self.dt * self.transition_scale * 100000
            self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces),
                                                    gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

        # set ShadowHand joint angles: [0,  1,  2,  4,  5,  6,  8,  9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21]
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

    # post_physics_step: compute observations and reward
    def post_physics_step(self):
        # get unpose quat 
        self.get_unpose_quat()
        # update buffer
        self.progress_buf += 1
        self.randomize_buf += 1
        # compute observation and reward
        if self.config['Modes']['train_default']: self.compute_observations_default()
        else: self.compute_observations()
        self.compute_reward(self.actions, self.id)

        # draw axes on target object
        if self.viewer:
            if self.debug_viz:
                self.gym.clear_lines(self.viewer)
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                for i in range(self.num_envs):
                    self.add_debug_lines(self.envs[i], self.object_pos[i], self.object_rot[i])
        
        # render and save each env camera view
        if self.render_each_view or self.render_hyper_view or self.sample_point_clouds:
            # init save render folder
            if self.render_folder is None:
                # create render_folder within logs/.../{test_*, train_*}
                if not self.is_testing: self.render_folder = osp.join(self.log_dir, 'train_{}'.format(len(glob.glob(osp.join(self.log_dir, 'train_*')))))
                elif self.is_testing and not self.config['Save']: self.render_folder = osp.join(self.log_dir, 'test_{}'.format(len(glob.glob(osp.join(self.log_dir, 'test_*')))))
                #make render_folder and save env_object_scale
                if self.render_folder is not None: 
                    os.makedirs(self.render_folder, exist_ok=True)
                    save_list_strings(os.path.join(self.render_folder, 'env_object_scale.txt'), self.env_object_scale)

            # locate frame, render with sampled fps
            test_sample, train_sample = 10, 8000 * 2
            if self.sample_point_clouds: test_sample = 1
            test_render_flag = self.frame % test_sample == 0 or self.frame == self.max_episode_length - 2
            train_render_flag = 0 <= self.frame % train_sample <= 300 and self.frame // train_sample >= 1
            if (self.is_testing and test_render_flag) or (not self.is_testing and train_render_flag):
                # start access image sensors
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)
                # render grid_camera_images (n_env, n_env) with (n_cam, n_cam)
                self.grid_camera_images, self.rendered_points, others = self.render_camera_images_points(render_images=self.render_each_view, sample_points=self.sample_point_clouds)
                # end access image tensors
                self.gym.end_access_image_tensors(self.sim)
                
                # save depth, rgb, seg images
                if self.render_each_view or self.render_hyper_view:
                    for key, value in self.grid_camera_images.items():
                        # # only save rgb images for visualization
                        if self.render_hyper_view and key not in ['rgb']: continue
                        save_path = osp.join(self.render_folder, '{}_{:03d}_{:03d}.png'.format(key, self.current_iteration, self.frame))
                        if not osp.exists(save_path): save_image(save_path, self.grid_camera_images[key].cpu().numpy().astype(np.uint8))
                
        # TODO: pytorch render images and points
        if self.config['Save_Render'] or self.sample_point_clouds:
            # pytorch render scene images(Nenv, Nview, H, W, RGBMDS) and points(Nenv, Npoint, XYZS)
            self.pytorch_rendered_images, self.pytorch_rendered_points = self.render_pytorch_images_points(self.hand_object_states, render_images=True, sample_points=True)
            # sample rendered object_points
            self.rendered_object_points, self.rendered_object_points_appears = sample_label_points(self.pytorch_rendered_points, label=SEGMENT_ID['object'][0], number=1024)
            self.rendered_object_points = self.rendered_object_points[..., :3]
            # compute rendered object_points centers
            self.rendered_object_points_centers = torch.mean(self.rendered_object_points, dim=1)
            # compute rendered object_features
            self.rendered_points_visual_features, _ = self.object_visual_encoder((self.rendered_object_points - self.rendered_object_points_centers.unsqueeze(1)).permute(0, 2, 1))
            self.rendered_points_visual_features = ((self.rendered_points_visual_features.squeeze(-1) - self.object_visual_scaler_mean) / self.object_visual_scaler_scale).float()
            # compute hand_object distances
            self.rendered_hand_object_dists = batch_sided_distance(self.hand_body_pos, self.rendered_object_points)
            # compute object pca axes
            self.rendered_object_pcas = torch.tensor(batch_decompose_pcas(self.rendered_object_points), device=self.device)

        # update root_state_tensor for object points for visualization
        if self.render_point_clouds:
            # set hand point clouds
            hand_points = torch.cat([self.hand_body_pos, torch.zeros((self.hand_body_pos.shape[0], self.hand_point_nums - self.hand_body_pos.shape[1], 3)).to(self.device)], dim=1)
            self.root_state_tensor[self.hand_point_indices, 0:3] = hand_points.reshape(-1, 3)
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(self.hand_point_indices.to(torch.int32)), len(self.hand_point_indices))
            # set rendered point clouds
            if self.sample_point_clouds:
                # self.root_state_tensor[self.object_point_indices, 0:3] = self.pytorch_rendered_points[..., :3].reshape(-1, 3)
                # self.root_state_tensor[self.object_point_indices, 0:3] = self.pytorch_rendered_points[..., (self.frame%3)*self.num_pc_downsample:(self.frame%3+1)*self.num_pc_downsample, :3].reshape(-1, 3)
                if self.frame % 3 == 0: render_points = self.pytorch_rendered_points[:, :1024, :]
                elif self.frame % 3 == 1: render_points, _ = sample_label_points(self.pytorch_rendered_points, SEGMENT_ID['hand'][0], 1024)
                elif self.frame % 3 == 2: render_points, _ = sample_label_points(self.pytorch_rendered_points, SEGMENT_ID['object'][0], 1024)
                self.root_state_tensor[self.object_point_indices, 0:3] = render_points[..., :3].reshape(-1, 3)
                self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(self.object_point_indices.to(torch.int32)), len(self.object_point_indices))
            # set object point clouds
            else:
                self.root_state_tensor[self.object_point_indices, 0:3] = self.object_points.reshape(-1, 3)
                self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(self.object_point_indices.to(torch.int32)), len(self.object_point_indices))


    # # ---------------------- Compute Reward ---------------------- # #
    # comute reward from actions
    def compute_reward(self, actions, id=-1):
        # compute hand reward
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.current_successes[:], self.consecutive_successes[:], self.final_successes[:] = compute_hand_reward(
            self.config['Modes'], self.config['Weights'],
            self.object_init_z, self.delta_qpos, self.delta_target_hand_pos, self.delta_target_hand_rot,
            self.id, self.object_id_buf, self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, 
            self.successes, self.current_successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot, self.goal_pos, self.goal_rot,
            self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, 
            self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, 
            self.fall_penalty, self.max_consecutive_successes, self.av_factor, self.goal_cond,
            # New obervations for computing reward
            self.object_points, self.right_hand_pc_dist, self.right_hand_finger_pc_dist, self.right_hand_joint_pc_dist, self.right_hand_body_pc_dist, self.delta_target_hand_pca
        )
        # append successes
        self.extras['successes'] = self.successes
        self.extras['current_successes'] = self.current_successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        # print success rate
        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()
            # The direct average shows the overall result more quickly, but slightly undershoots long term policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes / self.total_resets))


    # # ---------------------- Compute Observations ---------------------- # #
    # compute current observations
    def compute_observations(self):
        # refresh state tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # refresh force tensors
        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        # update object states
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_handle_pos = self.object_pos  ##+ quat_apply(self.object_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.06)
        self.object_back_pos = self.object_pos + quat_apply(self.object_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]
        
        # update object points (nenv, 1024, 3)
        self.object_points = batch_quat_apply(self.object_rot, self.object_init_mesh['points']) + self.object_pos.unsqueeze(1)
        self.object_points_centered = batch_quat_apply(self.object_rot, self.object_init_mesh['points_centered'])
        # encode dynamic object visual features
        if self.use_dynamic_visual_feats or self.config['Save']:
            with torch.no_grad():
                self.object_points_visual_features, _ = self.object_visual_encoder(self.object_points_centered.permute(0, 2, 1))
                self.object_points_visual_features = ((self.object_points_visual_features.squeeze(-1) - self.object_visual_scaler_mean) / self.object_visual_scaler_scale).float()
        # compute quat from hand_rot to object_pca
        self.object_pcas, self.target_hand_pca_rot = compute_hand_to_object_pca_quat(self.object_init_mesh['pca_axes'], self.object_rot, self.hand_prior_rot_quat)
        
        # right hand palm base
        idx = self.hand_body_idx_dict['palm']
        self.right_hand_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)
        
        # set finger shift
        self.finger_shift = 0.02
        if 'half_finger_shift' in self.config['Modes'] and self.config['Modes']['half_finger_shift']: self.finger_shift = 0.01
        # right hand fingertip body: index, middle, ring, little, thumb
        idx = self.hand_body_idx_dict['index']
        self.right_hand_ff_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.finger_shift)
        idx = self.hand_body_idx_dict['middle']
        self.right_hand_mf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.finger_shift)
        idx = self.hand_body_idx_dict['ring']
        self.right_hand_rf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.finger_shift)
        idx = self.hand_body_idx_dict['little']
        self.right_hand_lf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_lf_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.finger_shift)
        idx = self.hand_body_idx_dict['thumb']
        self.right_hand_th_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.finger_shift)
        # right hand fingertip joint
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        
        # update hand_joint_pos and hand_joint_rot (nenv, 17, 3)
        self.hand_joint_pos = self.rigid_body_states[:, self.valid_shadow_hand_bodies, 0:3]
        self.hand_joint_rot = self.rigid_body_states[:, self.valid_shadow_hand_bodies, 3:7]
        # update hand_body_pos (nenv, 36, 3)
        self.hand_body_pos = compute_hand_body_pos(self.hand_joint_pos, self.hand_joint_rot)

        # update goal pose
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        def world2obj_vec(vec):
            return quat_apply(quat_conjugate(self.object_rot), vec - self.object_pos)
        def obj2world_vec(vec):
            return quat_apply(self.object_rot, vec) + self.object_pos
        def world2obj_quat(quat):
            return quat_mul(quat_conjugate(self.object_rot), quat)
        def obj2world_quat(quat):
            return quat_mul(self.object_rot, quat)

        # Get hand dof pose
        self.dof_pos = self.shadow_hand_dof_pos
        # Distance from current hand pose to target hand pose
        self.delta_target_hand_pos = world2obj_vec(self.right_hand_pos) - self.target_hand_pos
        self.rel_hand_rot = world2obj_quat(self.right_hand_rot)
        self.delta_target_hand_rot = quat_mul(self.rel_hand_rot, quat_conjugate(self.target_hand_rot))
        self.delta_qpos = self.shadow_hand_dof_pos - self.target_qpos

        # Distance from hand pos to object point clouds
        self.right_hand_pc_dist = batch_sided_distance(self.right_hand_pos.unsqueeze(1), self.object_points).squeeze(-1)        
        self.right_hand_pc_dist = torch.where(self.right_hand_pc_dist >= 0.5, 0.5 + 0 * self.right_hand_pc_dist, self.right_hand_pc_dist)
        # Distance from hand finger pos to object point clouds
        self.right_hand_finger_pos = torch.stack([self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos], dim=1)
        self.right_hand_finger_pc_dist = torch.sum(batch_sided_distance(self.right_hand_finger_pos, self.object_points), dim=-1)
        self.right_hand_finger_pc_dist = torch.where(self.right_hand_finger_pc_dist >= 3.0, 3.0 + 0 * self.right_hand_finger_pc_dist, self.right_hand_finger_pc_dist)
        # Distance from all hand joint pos to object point clouds
        self.right_hand_joint_pc_dist = torch.sum(batch_sided_distance(self.hand_joint_pos, self.object_points), dim=-1) * 5 / self.hand_joint_pos.shape[1]
        self.right_hand_joint_pc_dist = torch.where(self.right_hand_joint_pc_dist >= 3.0, 3.0 + 0 * self.right_hand_joint_pc_dist, self.right_hand_joint_pc_dist)
        # Distance from all hand body pos to object point clouds
        self.right_hand_body_pc_batch_dist = batch_sided_distance(self.hand_body_pos, self.object_points)
        self.right_hand_body_pc_dist = torch.sum(self.right_hand_body_pc_batch_dist, dim=-1) * 5 / self.hand_body_pos.shape[1]
        self.right_hand_body_pc_dist = torch.where(self.right_hand_body_pc_dist >= 3.0, 3.0 + 0 * self.right_hand_body_pc_dist, self.right_hand_body_pc_dist)
        # Distance from hand rot to target pca rot
        self.delta_target_hand_pca = 2 * torch.acos(torch.abs(torch.clamp(torch.sum(self.right_hand_rot * self.target_hand_pca_rot, dim=1), -1.0, 1.0)))

        # compute current hand and object states
        self.hand_object_states = self.compute_hand_object_states()

        # vision_based setting
        if self.vision_based:
            # pytorch render scene images(Nenv, Nview, H, W, RGBMDS) and points(Nenv, Npoint, XYZS)
            self.pytorch_rendered_images, self.pytorch_rendered_points = self.render_pytorch_images_points(self.hand_object_states, render_images=True, sample_points=True)
            
            # save rendered depth images
            if self.render_folder is not None and self.frame % 10 == 0 and self.num_envs <= 10:
                # grid pytorch_rendered_images
                grid_pytorch_rendered_images = grid_camera_images(self.pytorch_rendered_images[:9, 0], size=[3, 3], border=False)
                # save grid depth images
                grid_depth_images = torch.where(-grid_pytorch_rendered_images[..., -2] > 1., 1., -grid_pytorch_rendered_images[..., -2]) * 255.
                save_path = osp.join(self.render_folder, '{}_{:03d}_{:03d}.png'.format('render_depth', self.current_iteration, self.frame))
                save_image(save_path, grid_depth_images.cpu().numpy().astype(np.uint8))

            # sample rendered object_points
            self.rendered_object_points, appears = sample_label_points(self.pytorch_rendered_points, label=SEGMENT_ID['object'][0], number=1024)
            self.rendered_object_points = self.rendered_object_points[..., :3]
            # compute rendered object_points centers
            self.rendered_object_points_centers = torch.mean(self.rendered_object_points, dim=1)
            # compute rendered object_features
            self.rendered_points_visual_features, _ = self.object_visual_encoder((self.rendered_object_points - self.rendered_object_points_centers.unsqueeze(1)).permute(0, 2, 1))
            self.rendered_points_visual_features = ((self.rendered_points_visual_features.squeeze(-1) - self.object_visual_scaler_mean) / self.object_visual_scaler_scale).float()
            # compute hand_object distances
            self.rendered_hand_object_dists = batch_sided_distance(self.hand_body_pos, self.rendered_object_points)

            # init vision_based_tracker
            if self.vision_based_tracker is None:
                self.vision_based_tracker = {'object_points': self.rendered_object_points.clone(),
                                             'object_centers': self.rendered_object_points_centers.clone(),
                                             'object_features': self.rendered_points_visual_features.clone(),
                                             'hand_object_dists': self.rendered_hand_object_dists.clone()}
                # init object_velocities with zero
                if 'use_object_velocities' in self.config['Distills'] and self.config['Distills']['use_object_velocities']:
                    self.vision_based_tracker['object_velocities'] = self.vision_based_tracker['object_centers'] - self.vision_based_tracker['object_centers']
                # init object_pcas
                if 'use_object_pcas' in self.config['Distills'] and self.config['Distills']['use_object_pcas']:
                    self.vision_based_tracker['object_pcas'] = torch.tensor(batch_decompose_pcas(self.rendered_object_points), device=self.device)
            
            # update object_velocities
            if 'use_object_velocities' in self.config['Distills'] and self.config['Distills']['use_object_velocities']:
                self.vision_based_tracker['object_velocities'][appears.squeeze(-1)==1] = (self.rendered_object_points_centers - self.vision_based_tracker['object_centers'])[appears.squeeze(-1)==1].clone()
            # update object_pcas: use_dynamic_pca
            if 'use_object_pcas' in self.config['Distills'] and self.config['Distills']['use_object_pcas']:
                if self.config['Distills']['use_dynamic_pcas']: self.vision_based_tracker['object_pcas'][appears.squeeze(-1)==1] = torch.tensor(batch_decompose_pcas(self.rendered_object_points), device=self.device)[appears.squeeze(-1)==1]

            # update vision_based_tracker with appeared object values
            self.vision_based_tracker['object_points'][appears.squeeze(-1)==1] = self.rendered_object_points[appears.squeeze(-1)==1].clone()
            self.vision_based_tracker['object_centers'][appears.squeeze(-1)==1] = self.rendered_object_points_centers[appears.squeeze(-1)==1].clone()
            self.vision_based_tracker['object_features'][appears.squeeze(-1)==1] = self.rendered_points_visual_features[appears.squeeze(-1)==1].clone()
            self.vision_based_tracker['hand_object_dists'][appears.squeeze(-1)==1] = self.rendered_hand_object_dists[appears.squeeze(-1)==1].clone()

        # compute full_state
        self.compute_full_state()
        if self.asymmetric_obs: self.compute_full_state(True)

    # # ---------------------- Compute Full State: ShadowHand and Object Pose ---------------------- # #
    def get_unpose_quat(self):
        if self.repose_z:
            self.unpose_z_theta_quat = quat_from_euler_xyz(torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta), -self.z_theta)
        return

    def unpose_point(self, point):
        if self.repose_z:
            return self.unpose_vec(point)
            # return self.origin + self.unpose_vec(point - self.origin)
        return point

    def unpose_vec(self, vec):
        if self.repose_z:
            return quat_apply(self.unpose_z_theta_quat, vec)
        return vec

    def unpose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.unpose_z_theta_quat, quat)
        return quat

    def unpose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.unpose_point(state[:, 0:3])
            state[:, 3:7] = self.unpose_quat(state[:, 3:7])
            state[:, 7:10] = self.unpose_vec(state[:, 7:10])
            state[:, 10:13] = self.unpose_vec(state[:, 10:13])
        return state
    
    def unpose_pc(self, pc):
        if self.repose_z:
            num_pts = pc.shape[1]
            return quat_apply(self.unpose_z_theta_quat.view(-1, 1, 4).expand(-1, num_pts, 4), pc)
        return pc

    def get_pose_quat(self):
        if self.repose_z:
            self.pose_z_theta_quat = quat_from_euler_xyz(torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta), self.z_theta)
        return

    def pose_vec(self, vec):
        if self.repose_z:
            return quat_apply(self.pose_z_theta_quat, vec)
        return vec

    def pose_point(self, point):
        if self.repose_z:
            return self.pose_vec(point)
            # return self.origin + self.pose_vec(point - self.origin)
        return point

    def pose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.pose_z_theta_quat, quat)
        return quat

    def pose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.pose_point(state[:, 0:3])
            state[:, 3:7] = self.pose_quat(state[:, 3:7])
            state[:, 7:10] = self.pose_vec(state[:, 7:10])
            state[:, 10:13] = self.pose_vec(state[:, 10:13])
        return state

    # compute current hand and object states
    def compute_hand_object_states(self):
        # get current hand_pose
        hand_pose = self.shadow_hand_dof_pos
        # get and unpack current hand_pos, hand_rot        
        hand_pos = self.unpose_point(self.hand_positions[self.hand_indices])
        hand_rot = self.unpose_quat(self.hand_orientations[self.hand_indices, :])
        # get current object_pos, object_rot
        object_pos = self.object_pos
        object_rot = self.object_rot
        # pack hand_object_state: [3, 4, 22, 3, 4]
        return torch.cat([hand_pos, hand_rot, hand_pose, object_pos, object_rot], dim=-1)
    
    # compute full observation state
    def compute_full_state(self, asymm_obs=False):
        # get unpose quat 
        self.get_unpose_quat()
        # unscale to (-1，1)
        num_ft_states = 13 * int(self.num_fingertips)  # 65 ##
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 30 ##

        # init obs dict
        obs_dict = dict()
        # # ---------------------- ShadowHand Observation 167 ---------------------- # #
        # 0:66, 22x3 shadow_hand dof positions, velocities, and forces
        hand_dof_pos = unscale(self.shadow_hand_dof_pos, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        hand_dof_vel = self.vel_obs_scale * self.shadow_hand_dof_vel
        hand_dof_force = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]
        obs_dict['hand_dofs'] = torch.cat([hand_dof_pos, hand_dof_vel, hand_dof_force], dim=-1)
        
        # 66:131, 13x5 shadow_hand finger position, orientation, linear and angular velocities
        aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        for i in range(5): aux[:, i * 13:(i + 1) * 13] = self.unpose_state(aux[:, i * 13:(i + 1) * 13])
        # 131:161: 6x5 shadow_hand finger force and torques, do not need repose
        finger_force_torques = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :30]
        obs_dict['hand_fingers'] = torch.cat([aux, finger_force_torques], dim=-1)
        
        # 161:167: 3+3 shadow_hand position, orientation
        hand_pos = self.unpose_point(self.right_hand_pos)
        hand_euler_xyz = get_euler_xyz(self.unpose_quat(self.hand_orientations[self.hand_indices, :]))
        obs_dict['hand_states'] = torch.cat([hand_pos, hand_euler_xyz[0].unsqueeze(-1), hand_euler_xyz[1].unsqueeze(-1), hand_euler_xyz[2].unsqueeze(-1)], dim=-1)

        # # ---------------------- Action Observation 24 ---------------------- # #
        # 167:191: action
        self.actions[:, 0:3] = self.unpose_vec(self.actions[:, 0:3])
        self.actions[:, 3:6] = self.unpose_vec(self.actions[:, 3:6])
        obs_dict['actions'] = self.actions

        # # ---------------------- Object Observation 16 / 25 ---------------------- # #
        # 191:207 object pos, rot, linvel, angvel
        object_pos = self.unpose_point(self.object_pose[:, 0:3])  # 3
        object_rot = self.unpose_quat(self.object_pose[:, 3:7])  # 4
        object_linvel = self.unpose_vec(self.object_linvel)  # 3
        object_angvel = self.vel_obs_scale * self.unpose_vec(self.object_angvel)  # 4
        object_hand_dist = self.unpose_vec(self.goal_pos - self.object_pos)  # 3
        obs_dict['objects'] = torch.cat([object_pos, object_rot, object_linvel, object_angvel, object_hand_dist], dim=-1)
        
        # encode obj_pca, append object_pca at the end
        if 'encode_obj_pca' in self.config['Modes'] and self.config['Modes']['encode_obj_pca']:
            obs_dict['objects'] = torch.cat([obs_dict['objects'], self.object_pcas.reshape(self.num_envs, -1)], dim=-1)
        
        # zero_object_state
        if 'zero_object_state' in self.config['Modes'] and self.config['Modes']['zero_object_state']:
            obs_dict['objects'] = torch.zeros_like(obs_dict['objects'], device=self.device)

        # # ---------------------- Object Visual Observation 128 ---------------------- # #
        # 207:335 object visual feature, default 0
        obs_dict['object_visual'] = self.object_points_visual_features * 0
        # zero_object_visual_feature
        if self.algo == 'ppo' and 'zero_object_visual_feature' in self.config['Modes'] and self.config['Modes']['zero_object_visual_feature']:
            obs_dict['object_visual'] = torch.zeros_like(obs_dict['object_visual'], device=self.device)
        if self.algo == 'dagger_value' and 'zero_object_visual_feature' in self.config['Distills']  and self.config['Distills']['zero_object_visual_feature']:
            obs_dict['object_visual'] = torch.zeros_like(obs_dict['object_visual'], device=self.device)
        # encode dynamic object visual features
        if self.use_dynamic_visual_feats: obs_dict['object_visual'] = self.object_points_visual_features

        # # ---------------------- Time Observation 29 ---------------------- # #
        # 335:364 encode time vector
        if self.config['Modes']['encode_obs_time']:
            obs_dict['times'] = torch.cat([self.progress_buf.unsqueeze(-1), compute_time_encoding(self.progress_buf, 28)], dim=-1)
            
        # # ---------------------- Hand-Object Observation 36 ---------------------- # #
        # 364:400 encode hand object dist
        if 'encode_hand_object_dist' in self.config['Modes'] and self.config['Modes']['encode_hand_object_dist']:
            obs_dict['hand_objects'] = self.right_hand_body_pc_batch_dist
        
        # # ---------------------- Vision Based Setting ---------------------- # #
        # TODO: update vision_based observations
        if self.vision_based:
            # update objects with rendered object_centers
            obs_dict['objects'] *= 0.
            obs_dict['objects'][:, :3] = self.vision_based_tracker['object_centers']
            # update objects with estimated velocities
            if 'use_object_velocities' in self.config['Distills'] and self.config['Distills']['use_object_velocities']:
                obs_dict['objects'][:, 3:6] = self.vision_based_tracker['object_velocities']
            # update objects with estimated pcas
            if 'use_object_pcas' in self.config['Distills'] and self.config['Distills']['use_object_pcas']:
                obs_dict['objects'][:, 6:15] = self.vision_based_tracker['object_pcas'].reshape(self.vision_based_tracker['object_pcas'].shape[0], -1)
            # update object_visual with rendered object_features
            obs_dict['object_visual'] = self.vision_based_tracker['object_features']
            # update hand_objects with rendered hand_object_dists
            obs_dict['hand_objects'] = self.vision_based_tracker['hand_object_dists']

        # Make Final Obs List
        self.obs_names = ['hand_dofs', 'hand_fingers', 'hand_states', 'actions', 'objects', 'object_visual', 'times', 'hand_objects', 'object_ids', 'object_hots']
        # Cat Final Obs Buff
        self.obs_buf = torch.cat([obs_dict[name] for name in self.obs_names if name in obs_dict], dim=-1)

        # Make Final Obs Interval Dict
        start_temp, self.obs_infos = 0, {'names': [name for name in self.obs_names if name in obs_dict], 'intervals': {}}
        for name in self.obs_names:
            if name not in obs_dict: continue
            self.obs_infos['intervals'][name] = [start_temp, start_temp + obs_dict[name].shape[-1]]
            start_temp += obs_dict[name].shape[-1]
        # # Check obs_infos within config file
        # if 'Obs' in self.config: assert self.config['Obs']['names'] == self.obs_infos['names'] and self.config['Obs']['intervals'] == self.obs_infos['intervals'], "Wrong Obs names and intervals!"

        return

    # # ---------------------- Reset Environment ---------------------- # #
    # reset goal pose
    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)
        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        rand_length = torch_rand_float(0.3, 0.5, (len(env_ids), 1), device=self.device)
        rand_angle = torch_rand_float(-1.57, 1.57, (len(env_ids), 1), device=self.device)
        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]

        # self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3]  # + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    # reset envs with random object rotation
    def reset(self, env_ids, goal_env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # # random goal_prior
        # self.random_prior = False
        if self.random_prior:
            for env_id in env_ids:
                i = env_id.item()
                # get object name and scale
                object_code_scale = self.env_object_scale[i]
                object_code = '{}/{}'.format(object_code_scale.split('/')[0], object_code_scale.split('/')[1])
                scale_str = object_code_scale.split('/')[2]
                # get grasp pose for object_scale
                data = self.grasp_data[object_code][scale_str] # data for one object one scale
                buf = data['object_euler_xy']
                prior_idx = random.randint(0, len(buf) - 1)
                # prior_idx = 0 ## use only one data
                # randomly pick target grasp pose
                self.target_qpos[i:i+1] = data['target_qpos'][prior_idx]
                self.target_hand_pos[i:i + 1] = data['target_hand_pos'][prior_idx]
                self.target_hand_rot[i:i + 1] = data['target_hand_rot'][prior_idx]
                self.object_init_euler_xy[i:i + 1] = data['object_euler_xy'][prior_idx]
                self.object_init_z[i:i + 1] = data['object_init_z'][prior_idx]

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5 + self.num_shadow_hand_dofs]
        # set shadow_hand_default_dof_pos
        pos = self.shadow_hand_default_dof_pos  # + self.reset_dof_pos_noise * rand_delta
        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_shadow_hand_dofs:5 + self.num_shadow_hand_dofs * 2]

        # set previous and current hand joint targets as default: 22
        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

        # get hand_indices within all envs: [0, 4, 8, 12]
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.unique(torch.cat([hand_indices]).to(torch.int32))

        # set hand joint initial states
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        # set hand joint target positions
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))
        # set default root tensor
        all_indices = torch.unique(torch.cat([all_hand_indices, self.object_indices[env_ids], self.table_indices[env_ids], ]).to(torch.int32))  ##
        self.hand_positions[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 0:3]
        self.hand_orientations[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 3:7]

        # sample random object rotation
        theta = torch_rand_float(-3.14, 3.14, (len(env_ids), 1), device=self.device)[:, 0]
        if not self.random_prior: theta *= 0
        # reset obejct with random rotation
        new_object_rot = quat_from_euler_xyz(self.object_init_euler_xy[env_ids,0], self.object_init_euler_xy[env_ids,1], theta)
        prior_rot_z = get_euler_xyz(quat_mul(new_object_rot, self.target_hand_rot[env_ids]))[2]

        # coordinate transform according to theta(object)/ prior_rot_z(hand)
        self.z_theta[env_ids] = prior_rot_z
        prior_rot_quat = quat_from_euler_xyz(torch.tensor(1.57, device=self.device).repeat(len(env_ids), 1)[:, 0], torch.zeros_like(theta), prior_rot_z)

        self.hand_orientations[hand_indices.to(torch.long), :] = prior_rot_quat
        self.hand_linvels[hand_indices.to(torch.long), :] = 0
        self.hand_angvels[hand_indices.to(torch.long), :] = 0

        # record hand_prior_rot_quat for all hands
        if self.num_envs == len(env_ids):
            self.hand_prior_rot_quat = quat_from_euler_xyz(torch.tensor(1.57, device=self.device).repeat(self.num_envs, 1)[:, 0], torch.zeros_like(theta), prior_rot_z)

        # Compute quat from hand_rot to object_pca, Set hand target quaternion
        if self.config['Modes']['init_pca_hand']: _, self.hand_orientations[hand_indices.to(torch.long), :] = compute_hand_to_object_pca_quat(self.object_init_mesh['pca_axes'][env_ids], new_object_rot, prior_rot_quat)
        
        # static init object states
        if 'static_init' in self.config['Modes'] and self.config['Modes']['static_init']:
            # locate test_start_id
            test_start_id = 0 if self.num_envs == 1000 else self.num_envs * self.current_test_iteration
            # Save Trajectory Mode
            if self.config['Save']:
                if not self.config['Save_Train']: target_pos_rot = torch.stack([self.object_init_mesh['init_states_test'][id, id + test_start_id] for id in env_ids], dim=0)
                else: target_pos_rot = torch.stack([self.object_init_mesh['init_states_train'][id, random.randint(0, self.object_init_mesh['init_states_train'].shape[1]-1)] for id in env_ids], dim=0)
            # Train/Test Mode
            else:
                # assert test each initial state, assume all environments share one object
                if self.is_testing: target_pos_rot = torch.stack([self.object_init_mesh['init_states_test'][id, id + test_start_id] for id in env_ids], dim=0)
                # assert random train initial states
                else: target_pos_rot = torch.stack([self.object_init_mesh['init_states_train'][id, random.randint(0, self.object_init_mesh['init_states_train'].shape[1]-1)] for id in env_ids], dim=0)
            # ** reset object position and rotation
            target_pos_rot[:, 2] += 0.01  # prevent penetration
            if 'central_object' in self.config['Modes'] and self.config['Modes']['central_object']: target_pos_rot[:, :2] *= 0.  # central object
            self.root_state_tensor[self.object_indices[env_ids], :3] = target_pos_rot[:, :3].clone()
            self.root_state_tensor[self.object_indices[env_ids], 3:7] = target_pos_rot[:, 3:7].clone()  # reset object rotation
            self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])
        else:
            # ** reset object position and rotation
            self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
            self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot  # reset object rotation
            self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        all_indices = torch.unique(torch.cat([all_hand_indices,
                                              self.object_indices[env_ids],
                                              self.goal_object_indices[env_ids],
                                              self.table_indices[env_ids], ]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))

        if self.random_time:
            self.random_time = False
            self.progress_buf[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
        else:
            self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    modes: Dict[str, bool], weights: Dict[str, float],
    object_init_z, delta_qpos, delta_target_hand_pos, delta_target_hand_rot,
    id: int, object_id, dof_pos, rew_buf, reset_buf, reset_goal_buf, progress_buf,
    successes, current_successes, consecutive_successes,
    max_episode_length: float, object_pos, object_handle_pos, object_back_pos, object_rot, target_pos, target_rot,
    right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool,
    # New obervations for computing reward
    object_points, right_hand_pc_dist, right_hand_finger_pc_dist, right_hand_joint_pc_dist, right_hand_body_pc_dist, delta_target_hand_pca
):
    # # ---------------------- State Update  ---------------------- # #
    # Action penalty
    action_penalty = torch.sum(actions ** 2, dim=-1)
    # Object lowest and heighest surface point
    heighest = torch.max(object_points[:, :, -1], dim=1)[0]
    lowest = torch.min(object_points[:, :, -1], dim=1)[0]

    # # ---------------------- Target Initial Hand State ---------------------- # #
    # Assign target initial hand pos in the midair
    target_z = heighest + 0.05
    target_xy = object_pos[:, :2]
    target_init_pos = torch.cat([target_xy, target_z.unsqueeze(-1)], dim=-1)
    # Distance from hand pos to target axis
    right_hand_axis_dist = torch.norm(target_xy - right_hand_pos[:, :2], p=2, dim=-1)
    # Distance from hand pos to target height point
    right_hand_init_dist = torch.norm(target_init_pos - right_hand_pos, p=2, dim=-1)

    # Assign target initial hand pose in the midair
    target_init_pose = torch.tensor([0.1, 0., 0.6, 0., 0., 0., 0.6, 0., -0.1, 0., 0.6, 0., 0., -0.2, 0., 0.6, 0., 0., 1.2, 0., -0.2, 0.], dtype=dof_pos.dtype, device=dof_pos.device)
    delta_init_qpos_value = torch.norm(dof_pos - target_init_pose, p=1, dim=-1)


    # # ---------------------- Goal Distances ---------------------- # #
    # Distance from the object/hand pos to the goal pos
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    goal_hand_dist = torch.norm(target_pos - right_hand_pos, p=2, dim=-1)
    

    # # ---------------------- Hand Distances ---------------------- # #
    # # Distance from the hand pos to the object pos
    # right_hand_dist = torch.norm(object_handle_pos - right_hand_pos, p=2, dim=-1)
    # right_hand_dist = torch.where(right_hand_dist >= 0.5, 0.5 + 0 * right_hand_dist, right_hand_dist)
    # # Distance from the hand finger pos to object pos: ff, mf, rf, lf, th
    # right_hand_finger_dist = (torch.norm(object_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(
    #     object_handle_pos - right_hand_mf_pos, p=2, dim=-1) + torch.norm(object_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(
    #             object_handle_pos - right_hand_lf_pos, p=2, dim=-1) + torch.norm(object_handle_pos - right_hand_th_pos, p=2, dim=-1))
    # right_hand_finger_dist = torch.where(right_hand_finger_dist >= 3.0, 3.0 + 0 * right_hand_finger_dist, right_hand_finger_dist)

    # Replace hand_pos_dist with hand_pc_dist
    right_hand_dist = right_hand_pc_dist
    right_hand_body_dist = right_hand_body_pc_dist
    right_hand_joint_dist = right_hand_joint_pc_dist
    right_hand_finger_dist = right_hand_finger_pc_dist

    # # ---------------------- Reward Weights ---------------------- # #
    # unpack hyper params
    max_finger_dist, max_hand_dist, max_goal_dist = weights['max_finger_dist'], weights['max_hand_dist'], weights['max_goal_dist']

    # right_hand_body_pc_dist
    if 'right_hand_body_dist' not in weights: weights['right_hand_body_dist'] = 0.

    # # ---------------------- Reward Computing ---------------------- # #
    # goal_conditioned
    if not goal_cond:
        # # ---------------------- Hold Detection / Reward Before Hold ---------------------- # #
        # hold_flag: hand pos and finger reach object region
        hold_value = 2
        hold_flag = (right_hand_finger_dist <= max_finger_dist).int() + (right_hand_dist <= max_hand_dist).int()
        
        # flag_joint_dist: hold flag with all joint dist
        if 'flag_joint_dist' in modes and modes['flag_joint_dist']:
            hold_flag = (right_hand_joint_dist <= max_finger_dist).int() + (right_hand_dist <= max_hand_dist).int()

        # flag_body_dist: hold flag with all body dist
        if 'flag_body_dist' in modes and modes['flag_body_dist']:
            hold_flag = (right_hand_body_dist <= max_finger_dist).int() + (right_hand_dist <= max_hand_dist).int()

        # # encode_obs_time: hold_flag with time threshold
        # if modes['encode_obs_time']:
        #     hold_value = 3
        #     hold_flag += (progress_buf < 40).int()

        # # ---------------------- Hand Object Exploration ---------------------- # #
        object_points_sorted, _ = torch.sort(object_points, dim=-1)
        object_points_sorted = object_points_sorted[:, :object_points_sorted.shape[1]//4, :]
        random_indices = torch.randint(0, object_points_sorted.shape[1], (object_points_sorted.shape[0], 1))
        exploration_target_pos = object_points_sorted[torch.arange(object_points_sorted.shape[0]).unsqueeze(1), random_indices].squeeze(1)
        right_hand_exploration_dist = torch.norm(exploration_target_pos - right_hand_pos, p=2, dim=-1)

        # # ---------------------- Reward After Holding ---------------------- # #
        # Distanc from object pos to goal target pos
        goal_rew = torch.zeros_like(goal_dist)
        goal_rew = torch.where(hold_flag == hold_value, 1.0 * (0.9 - 2.0 * goal_dist), goal_rew)
        # Distance from hand pos to goal target pos
        hand_up = torch.zeros_like(goal_dist)
        hand_up = torch.where(lowest >= 0.61, torch.where(hold_flag == hold_value, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= 0.80, torch.where(hold_flag == hold_value, 0.2 - goal_hand_dist * 0 + weights['hand_up_goal_dist'] * (0.2 - goal_dist), hand_up), hand_up)
        # Already hold the object and Already reach the goal
        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(hold_flag == hold_value, torch.where(goal_dist <= max_goal_dist, 1.0 / (1 + 10 * goal_dist), bonus), bonus)
        
        # # ---------------------- Total Reward ---------------------- # #
        # init_reward: let hand approach inital height-axis point 
        init_reward = weights['delta_init_qpos_value'] * delta_init_qpos_value 
        init_reward += weights['right_hand_dist'] * right_hand_dist
        init_reward += weights['delta_target_hand_pca'] * delta_target_hand_pca 
        init_reward += weights['right_hand_exploration_dist'] * right_hand_exploration_dist 
        
        # grasp_reward: let hand fingers approach object, lift object to goal
        grasp_reward = weights['right_hand_body_dist'] * right_hand_body_dist + weights['right_hand_joint_dist'] * right_hand_joint_dist
        grasp_reward += weights['right_hand_finger_dist'] * right_hand_finger_dist + 2.0 * weights['right_hand_dist'] * right_hand_dist
        grasp_reward += weights['goal_dist'] * goal_dist + weights['goal_rew'] * goal_rew + weights['hand_up'] * hand_up + weights['bonus'] * bonus

        # Total Reward: init reward + grasp reward
        reward = torch.where(hold_flag != hold_value, init_reward, grasp_reward)
    
    else:
        # Difference between hand pose to target hand pose
        delta_hand_pos_value = torch.norm(delta_target_hand_pos, p=1, dim=-1)
        delta_hand_rot_value = 2.0 * torch.asin(torch.clamp(torch.norm(delta_target_hand_rot[:, 0:3], p=2, dim=-1), max=1.0))
        delta_qpos_value = torch.norm(delta_qpos, p=1, dim=-1)
        delta_value = 0.6 * delta_hand_pos_value + 0.04 * delta_hand_rot_value + 0.1 * delta_qpos_value 
        # Target flag: whether hand pose reaches the target hand pose
        target_flag = (delta_hand_pos_value <= 0.4).int() + (delta_hand_rot_value <= 1.0).int() + (delta_qpos_value <= 6.0).int()
        
        # Difference between object rotation and target rotation
        quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        # goal_hand_rew: already reached the target hand pose and hold the object, distance from object to goal target
        flag = (right_hand_finger_dist <= max_finger_dist).int() + (right_hand_dist <= max_hand_dist).int() + target_flag
        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        goal_hand_rew = torch.where(flag == 5, 1 * (0.9 - 2 * goal_dist), goal_hand_rew)
        # hand_up: already hold the object, distance from hand height to goal target height
        flag2 = (right_hand_finger_dist <= max_finger_dist).int() + (right_hand_dist <= max_hand_dist).int()
        hand_up = torch.zeros_like(right_hand_finger_dist)
        hand_up = torch.where(lowest >= 0.63, torch.where(flag2 == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= 0.80, torch.where(flag2 == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)
        # bonus: already reached the goal
        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(goal_dist <= max_goal_dist, 1.0 / (1 + 10 * goal_dist), bonus)
        # Total Reward: hand pose/finger to object, object to goal target, hand height to goal target height, goal reach bonus
        reward = - 0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus  - 0.5 * delta_value

    # Init reset_buff
    resets = reset_buf
    # Find out which envs hit the goal and update successes count
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    # Reset goal also
    goal_resets = resets
    # Compute successes: reach the goal during running
    successes = torch.where(goal_dist <= max_goal_dist, torch.ones_like(successes), successes)
    # Compute final_successes: reach the goal at the end
    final_successes = torch.where(goal_dist <= max_goal_dist, torch.ones_like(successes), torch.zeros_like(successes))
    # Compute current_successes: reach the episode length and reach the goal
    current_successes = torch.where(resets == 1, successes, current_successes)
    # Compute cons_successes
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, current_successes, cons_successes, final_successes

