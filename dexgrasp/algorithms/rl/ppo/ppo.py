import os, yaml
import numpy as np
import os.path as osp
import copy, statistics
import time, tqdm, glob, wandb

from turtle import done
from pickle import FALSE
from gym.spaces import Space
from collections import deque
from datetime import datetime
from matplotlib.patches import FancyArrow

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.general_utils import *
from algorithms.rl.ppo import RolloutStorage


class PPO:

    def __init__(self,
                 vec_env,
                 actor_critic_class,  # ActorCritic
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 init_noise_std=1.0,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=None,
                 model_cfg=None,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False,
                 is_vision = False
                 ):
        self.is_vision = is_vision
        self.config = vec_env.task.config

        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.device = device
        self.asymmetric = asymmetric

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.step_size = learning_rate

        # PPO components
        self.vec_env = vec_env
        self.actor_critic = actor_critic_class(self.observation_space.shape, self.state_space.shape, self.action_space.shape,
                                               init_noise_std, self.config['Models'], asymmetric=asymmetric, use_pc = self.is_vision)
        self.actor_critic.to(self.device)
        self.storage = RolloutStorage(self.vec_env.num_envs, num_transitions_per_env, self.observation_space.shape,
                                      self.state_space.shape, self.action_space.shape, self.device, sampler)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10) if self.print_log else None
        self.vec_env.task.log_dir = log_dir
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.save_traj = False # need to be modified
        self.current_learning_iteration = 0
        self.apply_reset = apply_reset
        

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path,map_location=self.device))
        #self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path,map_location=self.device))
        #self.actor_critic.load_state_dict(torch.load(path))
        self.current_learning_iteration = 0 #int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def run(self, num_learning_iterations, log_interval=1):
        # replace visual_feature for test set mode
        if self.is_testing and osp.exists(osp.join(self.log_dir, 'pc_feat.npy')):
            self.vec_env.task.config['Modes']['zero_object_visual_feature'] = False
            self.vec_env.task.visual_feat_buf = torch.tensor(np.load(osp.join(self.log_dir, 'pc_feat.npy')), device=self.device).repeat(self.vec_env.task.visual_feat_buf.shape[0], 1)

        id = -1
        # only random time for training
        if self.is_testing: self.vec_env.task.random_time = False
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()
        self.vec_env.task.is_testing = self.is_testing
        
        # save train/test config to log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        save_yaml(osp.join(self.log_dir, 'train.yaml'), self.vec_env.task.config)

        # test PPO model
        if self.is_testing:
            # save observation-action-value-success trajectory: save also render images
            if self.config['Save']:
                # results_train/****_seed0/; results_trajectory/****_seed0/trajectory/
                save_name = 'results_trajectory_train' if self.config['Save_Train'] else 'results_trajectory_test'
                self.vec_env.task.render_folder = osp.join(self.log_dir.replace('results_train', save_name), 'trajectory')
                # save rendered point clouds
                if self.vec_env.task.config['Save_Render']:
                    self.vec_env.task.render_folder = osp.join(self.log_dir.replace('results_train', 'results_trajectory_render'), 'trajectory')
                    self.vec_env.task.pointcloud_folder = osp.join(self.log_dir.replace('results_train', 'results_trajectory_render'), 'pointcloud')
                    os.makedirs(self.vec_env.task.pointcloud_folder, exist_ok=True)
            # always make test dir for testing mode: save render images
            elif self.vec_env.task.render_folder is None:
                self.vec_env.task.render_folder = osp.join(self.log_dir, 'test_{}'.format(len(glob.glob(osp.join(self.log_dir, 'test_*')))))
            # make folders for rendering
            os.makedirs(self.vec_env.task.render_folder, exist_ok=True)
            # save self.env_object_scale into txt
            save_list_strings(os.path.join(self.vec_env.task.render_folder, 'env_object_scale.txt'), self.vec_env.task.env_object_scale)
            
            # run test for test_iteration times
            final_success_rate = []
            for it in range(self.vec_env.task.test_iteration):
                start_time = time.time()
                self.vec_env.task.current_test_iteration = it + 1
                # Save rendered point cloud trajectory
                point_cloud_trajectory = {'canonical': [], 'rendered': [], 'centers': [], 'appears': [], 'features': [], 'pcas': [], 'hand_body': [], 'hand_object': []} if self.vec_env.task.config['Save'] and self.vec_env.task.config['Save_Render'] else None
                # Save observation-action-value-success trajectory
                obs_action_trajectory = {'observations': [], 'actions': [], 'values': [], 'states': [], 'features': [], 'successes': None, 'final_successes': None} if self.vec_env.task.config['Save'] else None
                # Test PPO for max_episode_length
                print('Testing iteration {}/{}//{}/{}'.format(self.log_dir.split('/')[-1], self.device, it, self.vec_env.task.test_iteration))
                # Save object initial state by simulating 10 steps
                if self.vec_env.task.config['Init']: self.vec_env.task.max_episode_length = 3
                for _ in range(self.vec_env.task.max_episode_length):
                    self.vec_env.task.frame = _
                    with torch.no_grad():
                        if self.apply_reset:
                            current_obs = self.vec_env.reset()
                        id = (id+1)%self.vec_env.task.max_episode_length
                        # Compute the action
                        actions, values = self.actor_critic.act_inference(current_obs, act_value=self.vec_env.task.config['Save'])
                        # Save observation-action-value-success trajectory
                        if obs_action_trajectory is not None: 
                            obs_action_trajectory['observations'].append(current_obs.clone())
                            obs_action_trajectory['actions'].append(actions.clone())
                            obs_action_trajectory['values'].append(values.clone())
                            obs_action_trajectory['states'].append(self.vec_env.task.hand_object_states.clone())
                            obs_action_trajectory['features'].append(self.vec_env.task.object_points_visual_features.clone())
                        # Save point cloud trajectory
                        if point_cloud_trajectory is not None:
                            point_cloud_trajectory['canonical'].append(self.vec_env.task.object_points.clone().to(torch.float16))
                            point_cloud_trajectory['rendered'].append(self.vec_env.task.rendered_object_points.clone().to(torch.float16))
                            point_cloud_trajectory['centers'].append(self.vec_env.task.rendered_object_points_centers.clone())
                            point_cloud_trajectory['appears'].append(self.vec_env.task.rendered_object_points_appears.clone())
                            point_cloud_trajectory['features'].append(self.vec_env.task.rendered_points_visual_features.clone())
                            point_cloud_trajectory['pcas'].append(self.vec_env.task.rendered_object_pcas.clone())
                            point_cloud_trajectory['hand_body'].append(self.vec_env.task.hand_body_pos.clone())
                            point_cloud_trajectory['hand_object'].append(self.vec_env.task.rendered_hand_object_dists.clone())
                        # Step the vec_environment, update observation
                        next_obs, rews, dones, infos = self.vec_env.step(actions, id)
                        current_obs.copy_(next_obs)
                    if _ == self.vec_env.task.max_episode_length-2:
                        # save success_rate
                        success_rate=self.vec_env.task.successes.sum()/self.vec_env.num_envs
                        if obs_action_trajectory is not None: 
                            obs_action_trajectory['successes'] = self.vec_env.task.successes.clone().unsqueeze(-1)
                            obs_action_trajectory['final_successes'] = self.vec_env.task.final_successes.clone().unsqueeze(-1)
                self.vec_env.task.current_iteration += 1
                final_success_rate.append(success_rate.item())
                np.savetxt(os.path.join(self.vec_env.task.render_folder, 'final_success_rate.txt'), np.asarray(final_success_rate))
                # save observation-action-value-success trajectory: (Nenv, 200, Nobs), (Nenv, 200, Nact), (Nenv, 200, 1), (Nenv, 1)
                if obs_action_trajectory is not None:
                    obs_action_trajectory['observations'] = torch.stack(obs_action_trajectory['observations'], dim=1).cpu().numpy()
                    obs_action_trajectory['actions'] = torch.stack(obs_action_trajectory['actions'], dim=1).cpu().numpy()
                    obs_action_trajectory['values'] = torch.stack(obs_action_trajectory['values'], dim=1).cpu().numpy()
                    obs_action_trajectory['states'] = torch.stack(obs_action_trajectory['states'], dim=1).cpu().numpy()
                    obs_action_trajectory['features'] = torch.stack(obs_action_trajectory['features'], dim=1).cpu().numpy()
                    obs_action_trajectory['successes'] = obs_action_trajectory['successes'].cpu().numpy()
                    obs_action_trajectory['final_successes'] = obs_action_trajectory['final_successes'].cpu().numpy()
                    # compute trajectory valids from object_pos
                    obs_action_trajectory['valids'] = compute_trajectory_valids(obs_action_trajectory['observations'][:, :, 191:194])
                    # save obs_action_trajectory, split with group_size
                    group_size = 10
                    group_number = self.vec_env.task.num_envs // group_size
                    ncurrent = len(glob.glob(osp.join(self.vec_env.task.render_folder, 'trajectory_*.pkl')))
                    for ngroup in range(group_number):
                        sub_trajectory = {key: value[ngroup*group_size:(ngroup + 1)*group_size] for key, value in obs_action_trajectory.items()}
                        save_pickle(osp.join(self.vec_env.task.render_folder, 'trajectory_{:03d}.pkl'.format(ncurrent + ngroup)), sub_trajectory)
                # save point cloud trajectory: (Nenv, 200, 1024, 3), (Nenv, 200, 1024, 3), (Nenv, 200, 36, 3)
                if point_cloud_trajectory is not None:
                    point_cloud_trajectory['canonical'] = torch.stack(point_cloud_trajectory['canonical'], dim=1).cpu().numpy()
                    point_cloud_trajectory['rendered'] = torch.stack(point_cloud_trajectory['rendered'], dim=1).cpu().numpy()
                    point_cloud_trajectory['centers'] = torch.stack(point_cloud_trajectory['centers'], dim=1).cpu().numpy()
                    point_cloud_trajectory['appears'] = torch.stack(point_cloud_trajectory['appears'], dim=1).cpu().numpy()
                    point_cloud_trajectory['features'] = torch.stack(point_cloud_trajectory['features'], dim=1).cpu().numpy()
                    point_cloud_trajectory['pcas'] = torch.stack(point_cloud_trajectory['pcas'], dim=1).cpu().numpy()
                    point_cloud_trajectory['hand_body'] = torch.stack(point_cloud_trajectory['hand_body'], dim=1).cpu().numpy()
                    point_cloud_trajectory['hand_object'] = torch.stack(point_cloud_trajectory['hand_object'], dim=1).cpu().numpy()
                    point_cloud_trajectory['successes'] = obs_action_trajectory['successes']
                    point_cloud_trajectory['final_successes'] = obs_action_trajectory['final_successes']
                    point_cloud_trajectory['valids'] = obs_action_trajectory['valids']
                    # save point_cloud_trajectory, split with group_size
                    group_size = 10
                    group_number = self.vec_env.task.num_envs // group_size
                    ncurrent = len(glob.glob(osp.join(self.vec_env.task.pointcloud_folder, 'pointcloud_*.pkl')))
                    for ngroup in range(group_number):
                        sub_trajectory = {key: value[ngroup*group_size:(ngroup + 1)*group_size] for key, value in point_cloud_trajectory.items()}
                        save_pickle(osp.join(self.vec_env.task.pointcloud_folder, 'pointcloud_{:03d}.pkl'.format(ncurrent + ngroup)), sub_trajectory)
                print('process time:', (time.time() - start_time) / 60)
            # save final success_rate for all iterations
            print("Final success_rate: {}/{:.3f}/{}".format(self.log_dir.split('/')[-1], np.mean(final_success_rate), self.vec_env.task.test_iteration))
            exit()
        
        # train PPO within env  
        else:
            # save self.env_object_scale into txt
            save_list_strings(os.path.join(self.log_dir, 'env_object_scale.txt'), self.vec_env.task.env_object_scale)
            
            # init wandb
            if self.vec_env.task.init_wandb: wandb.init(project=self.log_dir.split('/')[-1])
            # init buffers
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            print_logs, reward_sum, episode_length = [], [], []
            # start training with num_learning_iterations
            start_time = time.time()
            self.vec_env.task.frame = -1
            for it in range(self.current_learning_iteration, num_learning_iterations):
                ep_infos = []
                start = time.time()
                process_time = (start - start_time) / 60
                self.vec_env.task.current_iteration += 1
                # Print log info
                if it % (max(1, num_learning_iterations // 100)) == 0:
                    total_time = (process_time / (it+1)) * num_learning_iterations
                    print_logs.append("Training iteration: {}/{}//{}/{}; Processed time:{}/{} mins; Left time: {:.2f} hours".format(
                        self.log_dir.split('/')[-1], self.device, it, num_learning_iterations, int(process_time), int(total_time), (total_time - process_time) / 60))
                    print(print_logs[-1])
                    save_list_strings(osp.join(self.log_dir, 'train.log'), print_logs)

                # Rollout = 8
                for _ in range(self.num_transitions_per_env):
                    self.vec_env.task.frame += 1
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()
                    id = (id+1)%self.vec_env.task.max_episode_length
                    # Compute the action
                    actions, actions_log_prob, values, mu, sigma = self.actor_critic.act(current_obs, current_states)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions, id)
                    next_states = self.vec_env.get_state()
                    # Record the transition
                    self.storage.add_transitions(current_obs, current_states, actions, rews, dones, values, actions_log_prob, mu, sigma)
                    current_obs.copy_(next_obs)
                    current_states.copy_(next_states)
                    # Book keeping
                    ep_infos.append(infos)

                    # Print log
                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                # Print log
                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                _, _, last_values, _, _ = self.actor_critic.act(current_obs, current_states)
                stop = time.time()
                collection_time = stop - start
                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                self.storage.compute_returns(last_values, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss = self.update()
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals(), show=False)
                if it % log_interval == 0 and it != 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                ep_infos.clear()
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))
            exit()

    def log(self, locs, width=80, pad=35, show=False):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
                # update wandb
                if key in ['successes', 'current_successes', 'consecutive_successes']:
                    if self.vec_env.task.init_wandb: wandb.log({'Mean episode ' + key: value})
        mean_std = self.actor_critic.log_std.exp().mean()

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        self.writer.add_scalar('Train2/mean_reward/step', locs['mean_reward'], locs['it'])
        self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
            if self.vec_env.task.init_wandb: wandb.log({'Mean reward': statistics.mean(locs['rewbuffer']), 'Mean reward per step': locs['mean_reward']})
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        if show: print(log_string)

    def update(self):
        # value, policy losses and batch
        mean_value_loss = 0
        mean_surrogate_loss = 0
        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        # update policy and value nets for epochs 
        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):

            # policy, value net batch update
            for indices in batch:
                # unpack batch observations, actions, target_values, returns, advantages
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                if self.asymmetric: states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                else: states_batch = None
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]

                # evaluate current policy
                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(obs_batch, states_batch, actions_batch)

                # KL, Step_Size Manager
                if self.desired_kl != None and self.schedule == 'adaptive':
                    kl = torch.sum(sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.step_size

                # Advantage loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # Total loss
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                # Loss log
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        return mean_value_loss, mean_surrogate_loss
