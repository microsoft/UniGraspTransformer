import os
import glob
import time
import tqdm
import yaml
import statistics
import numpy as np
import os.path as osp
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from gym import spaces
from gym.spaces import Space
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.general_utils import *
from algorithms.rl.dagger_value.storage import RolloutStorage, PPORolloutStorage, PERBuffer
from algorithms.rl.dagger_value.module import ActorCriticTransformerEncoder


class DAGGERVALUE:

    def __init__(self,
                 vec_env,
                 actor_class,  # Actor
                 actor_critic_class,  # ActorCriticDagger
                 actor_critic_class_expert,  # ActorCritic
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 buffer_size,
                 init_noise_std=1.0,
                 learning_rate=1e-3,
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
                 is_vision = False,
                 expert_chkpt_path = "",
                 ):
        # load dagger_value config
        self.is_vision = is_vision
        self.config = vec_env.task.config
        with open('cfg/dagger_value/config.yaml', 'r') as f: self.cfg = yaml.safe_load(f)

        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_shape = vec_env.observation_space.shape[0]
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.device = device
        self.asymmetric = asymmetric

        self.schedule = schedule
        self.desired_kl = desired_kl
        self.step_size = learning_rate

        # DAGGER parameters
        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.num_transitions_per_env = num_transitions_per_env
        # Update DAGGER parameters from Config
        if 'num_mini_batches' in self.config['Distills']: self.num_mini_batches = self.config['Distills']['num_mini_batches']
        if 'num_learning_epochs' in self.config['Distills']: self.num_learning_epochs = self.config['Distills']['num_learning_epochs']
        if 'num_transitions_per_env' in self.config['Distills']: self.num_transitions_per_env = self.config['Distills']['num_transitions_per_env']

        # DAGGER components
        self.vec_env = vec_env
        self.buffer_size = buffer_size
        # create actor_critic net
        self.value_loss_cfg = self.cfg['learn']['value_loss']
        self.apply_value_net = self.value_loss_cfg['apply']

        # Log
        self.log_dir = log_dir
        self.vec_env.task.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10) if self.print_log else None
        self.tot_time, self.tot_timesteps = 0, 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0
        self.apply_reset = apply_reset

        # Pick model type: dagger_value, transformer
        self.use_model_type = self.config['Distills']['use_model_type'] if 'use_model_type' in self.config['Distills'] else 'dagger_value'
        
        # Load transformer model: transformer for actor model and mlp for value model 
        if self.use_model_type == 'transformer_encoder':
            # init ActorCriticTransformerEncoder
            self.actor = ActorCriticTransformerEncoder(obs_shape=(self.config['Weights']['num_observation'], ), states_shape=None, actions_shape=(24, ), initial_std=0.8, model_cfg=self.config)
            self.actor.to(self.device)
            # init ActorCriticTransformerEncoder Actor Optimizer
            self.optimizer = optim.Adam(self.actor.actor.parameters(), lr=float(self.config['Offlines']['learning_rate']))
            self.optimizer_value = optim.Adam(self.actor.critic.parameters(), lr=learning_rate)
        # Load dagger_value model: mlps for actor and value models
        elif self.use_model_type == 'dagger_value':
            # init student and expert model_cfg
            student_model_cfg, expert_model_cfg = self.config['Models'].copy(), self.config['Models'].copy()
            # modify student model_cfg
            if 'pi_hid_sizes' in self.config['Distills'] and 'vf_hid_sizes' in self.config['Distills']:
                student_model_cfg['pi_hid_sizes'] = self.config['Distills']['pi_hid_sizes']
                student_model_cfg['vf_hid_sizes'] = self.config['Distills']['vf_hid_sizes']
                student_model_cfg['sigmoid_actions'] = self.config['Distills']['sigmoid_actions']

            # apply action-value network
            if self.apply_value_net:
                # create actor and critic net with seperate optimizer, optimizer_value
                self.actor = actor_critic_class(self.observation_space.shape, self.state_space.shape, self.action_space.shape,
                                                init_noise_std, student_model_cfg, asymmetric=asymmetric, use_pc=self.is_vision)
                self.optimizer = optim.Adam(self.actor.actor.parameters(), lr=learning_rate)
                self.optimizer_value = optim.Adam(self.actor.critic.parameters(), lr=learning_rate)
            # apply action network only
            else:
                self.actor = actor_class(self.observation_space.shape, self.state_space.shape, self.action_space.shape,
                                         init_noise_std, student_model_cfg, asymmetric=asymmetric, use_pc=self.is_vision)
                self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
            self.actor.to(self.device)

        # return for test mode, without creating storage and expert models
        if self.is_testing: return

        # create ppo storage buffer
        self.ppo_buffer = PPORolloutStorage(self.vec_env.num_envs, num_transitions_per_env, self.observation_space.shape,
                                            self.state_space.shape, self.action_space.shape, self.device, sampler)
        # create distill storage buffer
        self.storage = RolloutStorage(self.vec_env.num_envs, self.buffer_size, self.observation_space.shape,
                                      self.state_space.shape, self.action_space.shape, self.device, sampler)

        # create expert_observation_space
        self.is_vision_expert = False
        self.expert_observation_space = self.observation_space if self.is_vision_expert \
            else spaces.Box(np.ones(self.observation_shape) * -np.Inf, np.ones(self.observation_shape))
        # get env task info
        self.task = self.vec_env.task
        self.config = self.task.config
        self.num_envs = self.task.num_envs
        self.env_object_scale = self.task.env_object_scale
        self.object_line_list = self.task.object_line_list
        self.object_scale_list = self.task.object_scale_list
        # create expert_config for each object-expert checkpoint
        self.expert_list, self.expert_cfg_list = [], []
        for nline in range(len(self.object_line_list)):
            self.expert_cfg_list.append({
                'name': '{:04d}'.format(self.object_line_list[nline]),
                'path': os.path.join(self.config['Save_Base'], 'results_train', '{:04d}_seed0/model_10000.pt'.format(self.object_line_list[nline])),
                'object_scale_list': [self.object_scale_list[nline]]
            })

        # multi_expert loading
        for expert_id, expert_cfg in enumerate(self.expert_cfg_list):
            # create expert agent
            expert = actor_critic_class_expert(self.expert_observation_space.shape, self.state_space.shape, self.action_space.shape,
                                               init_noise_std, expert_model_cfg, asymmetric=asymmetric, use_pc=self.is_vision_expert)
            expert.to(self.device)
            expert.load_state_dict(torch.load(expert_cfg['path'], map_location=self.device))
            self.expert_list.append(expert)
            
            # update expert_cfg['indices']
            expert_cfg['indices'] = []
            for expert_object_scale in expert_cfg['object_scale_list']:
                # locate expert_object_scale in env_object_scale
                split_temp = expert_object_scale.split('/')
                expert_object_scale_name = '{}/{}/{}'.format(split_temp[0], split_temp[1], self.task.scale2str[float(split_temp[2])])
                expert_cfg['indices'] += [index for index, string in enumerate(self.env_object_scale) if expert_object_scale_name == string]
                print('Expert {}/{} covers num envs: {} {}~{}'.format(expert_id, expert_cfg['name'], len(expert_cfg['indices']), min(expert_cfg['indices']), max(expert_cfg['indices'])))
                print('Expert object_scale_list: {}'.format(expert_cfg['object_scale_list']))
                print('Expert checkpoint: {}'.format(expert_cfg['path']))
                # for id in expert_cfg['indices']: print('Expert env {}-{}'.format(id, self.env_object_scale[id]))

    def get_all_checkpoints_in_dir(self, workdir: str):
        model_dir_list = []
        for file in os.listdir(workdir):
            if file.endswith('.pt'):
                model_dir_list.append(osp.join(workdir, file))
        return model_dir_list

    def test(self, path):
        self.actor.load_state_dict(torch.load(path,map_location=self.device))
        self.actor.eval()

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        self.current_learning_iteration = 0 #int(path.split("_")[-1].split(".")[0])
        self.actor.train()

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    # multi_expert
    def expert_inference(self, current_obs, act_value=False):
        # assert zero_object_visual for PPO expert
        if 'zero_object_visual_feature' in self.config['Modes'] and self.config['Modes']['zero_object_visual_feature']:
            current_obs[:, self.task.obs_infos['intervals']['object_visual'][0]:self.task.obs_infos['intervals']['object_visual'][1]] *= 0.
        # take expert actions
        action = torch.zeros((current_obs.shape[0], self.action_space.shape[0]), device=self.device)
        value = torch.zeros((current_obs.shape[0], 1), device=self.device)
        for expert, expert_cfg in zip(self.expert_list, self.expert_cfg_list):
            indices = expert_cfg['indices']
            if act_value: action[indices], value[indices] = expert.act_inference(current_obs[indices], act_value=True)
            else: action[indices], _ = expert.act_inference(current_obs[indices], act_value=False)
        return action, value

    def run(self, num_learning_iterations, log_interval=1):
        id = -1
        if self.is_testing: self.vec_env.task.random_time = False
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()
        self.vec_env.task.is_testing = self.is_testing

        # save distill config to log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        save_yaml(osp.join(self.log_dir, 'distill.yaml'), self.vec_env.task.config)
        
        # test Dagger_Value Distilled Model
        if self.is_testing:
            # save observation-action-value-success trajectory: save also render images
            if self.config['Save']: 
                self.vec_env.task.render_folder = osp.join(self.log_dir, 'trajectory')
                os.makedirs(self.vec_env.task.render_folder, exist_ok=True)
            # always make test dir for testing mode
            if self.vec_env.task.render_folder is None:
                if self.vec_env.task.test_epoch != 0: self.vec_env.task.render_folder = osp.join(self.log_dir, 'test_model_{}'.format(self.vec_env.task.test_epoch))
                else: self.vec_env.task.render_folder = osp.join(self.log_dir, 'test_{}'.format(len(glob.glob(osp.join(self.log_dir, 'test_*')))))
                os.makedirs(self.vec_env.task.render_folder, exist_ok=True)
            # save self.env_object_scale into txt
            save_list_strings(os.path.join(self.vec_env.task.render_folder, 'env_object_scale.txt'), self.vec_env.task.env_object_scale)

            # run test for test_iteration times
            final_success_rate = []
            for it in range(self.vec_env.task.test_iteration):
                self.vec_env.task.current_test_iteration = it + 1
                print('Test iteration {}/{}//{}/{}'.format(self.log_dir.split('/')[-1], self.device, it, self.vec_env.task.test_iteration))
                # Save observation-action-value-success trajectory
                obs_action_trajectory = {'observations': [], 'actions': [], 'values': [], 'states': [], 'features': [], 'successes': None, 'final_successes': None} if self.config['Save'] else None
                for _ in range(self.vec_env.task.max_episode_length):
                    self.vec_env.task.frame = _
                    with torch.no_grad():
                        if self.apply_reset:
                            current_obs = self.vec_env.reset()
                            if self.apply_value_net: current_states = self.vec_env.get_state()
                        id = (id+1)%self.vec_env.task.max_episode_length
                        # Compute the action
                        actions, values = self.actor.act_inference(current_obs)
                        # Save observation-action-value-success trajectory
                        if obs_action_trajectory is not None: 
                            obs_action_trajectory['observations'].append(current_obs.clone())
                            obs_action_trajectory['actions'].append(actions.clone())
                            # obs_action_trajectory['values'].append(values.clone())
                            obs_action_trajectory['states'].append(self.vec_env.task.hand_object_states.clone())
                            obs_action_trajectory['features'].append(self.vec_env.task.object_points_visual_features.clone())
                        # Step the vec_environment
                        next_obs, rews, dones, infos = self.vec_env.step(actions,id)
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
                    # obs_action_trajectory['values'] = torch.stack(obs_action_trajectory['values'], dim=1).cpu().numpy()
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
            # save final success_rate for all iterations
            print("Final success_rate: {}/{:.3f}/{}".format(self.log_dir.split('/')[-1], np.mean(final_success_rate), self.vec_env.task.test_iteration))
            exit()
            
        else:
            # save self.env_object_name into txt
            save_list_strings(os.path.join(self.log_dir, 'env_object_scale.txt'), self.vec_env.task.env_object_scale)
            
            # init buffers
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            print_logs, reward_sum, episode_length = [], [], []
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
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
                    print_logs.append("Distilling iteration: {}/{}//{}/{}; Processed time:{}/{} mins; Left time: {:.2f} hours".format(
                        self.log_dir.split('/')[-1], self.device, it, num_learning_iterations, int(process_time), int(total_time), (total_time - process_time) / 60))
                    print(print_logs[-1])
                    save_list_strings(osp.join(self.log_dir, 'distill.log'), print_logs)

                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        # value_net
                        if self.apply_value_net:
                            current_states = self.vec_env.get_state()
                    id = (id+1)%self.vec_env.task.max_episode_length

                    # Compute the action
                    # value_net
                    if self.apply_value_net:
                        actions, actions_log_prob, values, mu, sigma = self.actor.act(current_obs, current_states)
                    else: actions, values = self.actor.act_inference(current_obs)
                    # multi_expert
                    actions_expert, values_expert = self.expert_inference(current_obs.clone(), act_value=False)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions, id)
                    next_states = self.vec_env.get_state()
                    # Record the transition
                    self.storage.add_transitions(current_obs, actions_expert, rews, dones)
                    current_obs.copy_(next_obs)
                    # value_net
                    if self.apply_value_net:
                        self.ppo_buffer.add_transitions(current_obs, current_states, actions, rews, dones, values, actions_log_prob, mu, sigma, actions_expert, values_expert)
                        current_states.copy_(next_states)
                    # Book keeping
                    ep_infos.append(infos)

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)
                
                # value_net
                if self.apply_value_net:
                    actions, actions_log_prob, values, mu, sigma = self.actor.act(current_obs, current_states)
                else: actions, values = self.actor.act_inference(current_obs)

                stop = time.time()
                collection_time = stop - start
                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                if self.apply_value_net:
                    self.ppo_buffer.compute_returns(values, self.value_loss_cfg['gamma'], self.value_loss_cfg['lam'])
                    mean_policy_loss, mean_value_loss = self.update()
                    self.ppo_buffer.clear()
                else:
                    mean_policy_loss = self.update()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if it % log_interval == 0 and it != 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                ep_infos.clear()
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))
            exit()

    def log(self, locs, width=80, pad=35):
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

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/policy', locs['mean_policy_loss'], locs['it'])

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
                          f"""{'Policy loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Policy loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def update(self):
        # combine update policy and value nets
        if 'combine_policy_value_net' in self.config['Distills'] and self.config['Distills']['combine_policy_value_net']:
            return self.combine_update()
        
        # policy loss and batch
        mean_policy_loss = 0
        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        # value loss and batch
        if self.apply_value_net:
            mean_value_loss = 0
            batch_value = self.ppo_buffer.mini_batch_generator(self.num_mini_batches)
        # update policy and value nets for epochs
        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):

            # policy net batch update
            for indices in batch:
                # unpack batch observation and action_expert
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                actions_expert_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                # actor with value_net
                if self.apply_value_net: actions_batch, values_batch = self.actor.act_withgrad(obs_batch, act_value=False)  # TODO: act with value net
                else: actions_batch = self.actor.act(obs_batch)

                # Policy loss
                dagger_loss = F.mse_loss(actions_batch, actions_expert_batch)  # 5 epochs x 5 batches x batch_size (500000, *), change 16000 per update
                # Gradient step
                self.optimizer.zero_grad()
                dagger_loss.backward()
                #nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()
                # update loss log
                mean_policy_loss += dagger_loss.item()

            # value net batch update
            if self.apply_value_net:
                for indices in batch_value:
                    # unpack batch observation and actions_batch, target_values_batch, returns_batch
                    obs_batch = self.ppo_buffer.observations.view(-1, *self.ppo_buffer.observations.size()[2:])[indices]
                    if self.asymmetric: states_batch = self.ppo_buffer.states.view(-1, *self.ppo_buffer.states.size()[2:])[indices]
                    else: states_batch = None
                    actions_batch = self.ppo_buffer.actions.view(-1, self.ppo_buffer.actions.size(-1))[indices]
                    target_values_batch = self.ppo_buffer.values.view(-1, 1)[indices]
                    returns_batch = self.ppo_buffer.returns.view(-1, 1)[indices]

                    # evaluate current policy
                    actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor.evaluate(obs_batch, states_batch, actions_batch)

                    # Value loss
                    if self.value_loss_cfg['use_clipped_value_loss']:
                        clip_range = self.value_loss_cfg['clip_range']
                        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-clip_range, clip_range)
                        value_losses = (value_batch - returns_batch).pow(2)
                        value_losses_clipped = (value_clipped - returns_batch).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = (returns_batch - value_batch).pow(2).mean()
                    value_loss = value_loss * self.value_loss_cfg['value_loss_coef']
                    # Gradient step
                    self.optimizer_value.zero_grad()
                    value_loss.backward()
                    #nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.optimizer_value.step()
                    # update loss log
                    mean_value_loss += value_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_policy_loss /= num_updates
        if self.apply_value_net:
            mean_value_loss /= num_updates
            return mean_policy_loss, mean_value_loss
        else: return mean_policy_loss


    def combine_update(self):
        # hyper_params
        self.clip_param = 0.2
        self.max_grad_norm = 1.0
        # value, policy losses and batch
        mean_policy_loss, mean_value_loss = 0, 0
        batch = self.ppo_buffer.mini_batch_generator(self.num_mini_batches)
        batch_expert = self.storage.mini_batch_generator(self.num_mini_batches)
        # update policy and value nets for epochs
        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):

            # expert, policy, value batch update
            for indices, expert_indices in zip(batch, batch_expert):
                # unpack batch observation and action_expert
                obs_storage_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[expert_indices]
                actions_expert_storage_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[expert_indices]
                # values_expert_storage_batch = self.storage.actions.view(-1, self.storage.values.size(-1))[expert_indices]
                actions_current_storage_batch, values_current_storage_batch = self.actor.act_withgrad(obs_storage_batch, act_value=False)  # TODO: act with value net
                # Dagger action loss
                dagger_action_loss = F.mse_loss(actions_current_storage_batch, actions_expert_storage_batch)

                # Unpack ppo_storage batch: current_obs, actions_expert, values_expert
                obs_batch = self.ppo_buffer.observations.view(-1, *self.ppo_buffer.observations.size()[2:])[indices]
                actions_expert_batch = self.ppo_buffer.actions_expert.view(-1, self.ppo_buffer.actions_expert.size(-1))[indices]
                # values_expert_batch = self.ppo_buffer.values_expert.view(-1, self.ppo_buffer.values_expert.size(-1))[indices]
                actions_current_batch, values_current_batch = self.actor.act_withgrad(obs_batch, act_value=False)  # TODO: act with value net
                # Expert action loss
                expert_action_loss = F.mse_loss(actions_current_batch, actions_expert_batch)

                # Unpack ppo_storage batch: actions, values, returns, advantages, actions_log_prob, mu, sigma
                if self.asymmetric: states_batch = self.ppo_buffer.states.view(-1, *self.ppo_buffer.states.size()[2:])[indices]
                else: states_batch = None
                actions_batch = self.ppo_buffer.actions.view(-1, self.ppo_buffer.actions.size(-1))[indices]
                target_values_batch = self.ppo_buffer.values.view(-1, 1)[indices]
                returns_batch = self.ppo_buffer.returns.view(-1, 1)[indices]
                advantages_batch = self.ppo_buffer.advantages.view(-1, 1)[indices]
                old_mu_batch = self.ppo_buffer.mu.view(-1, self.ppo_buffer.actions.size(-1))[indices]
                old_sigma_batch = self.ppo_buffer.sigma.view(-1, self.ppo_buffer.actions.size(-1))[indices]
                old_actions_log_prob_batch = self.ppo_buffer.actions_log_prob.view(-1, 1)[indices]

                # evaluate current policy
                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor.evaluate(obs_batch, states_batch, actions_batch)

                # KL, Step_Size Manager
                self.desired_kl = None  # # TODO: turn on desired_kl
                if self.desired_kl != None and self.schedule == 'adaptive':
                    kl = torch.sum(sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)
                    for param_group in self.optimizer_combine.param_groups:
                        param_group['lr'] = self.step_size

                # Action loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                ppo_action_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value loss
                if self.config['Distills']['use_clipped_value_loss']:
                    clip_range = self.value_loss_cfg['clip_range']
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-clip_range, clip_range)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (value_batch - returns_batch).pow(2).mean()
                ppo_value_loss = value_loss

                # Total loss
                loss = self.config['Distills']['dagger_action_loss'] * dagger_action_loss + self.config['Distills']['expert_action_loss'] * expert_action_loss \
                    + self.config['Distills']['ppo_action_loss'] * ppo_action_loss + self.config['Distills']['ppo_value_loss'] * ppo_value_loss + self.config['Distills']['ppo_entropy_loss'] * entropy_batch.mean()
                
                # Gradient step
                self.optimizer_combine.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer_combine.step()
                # Loss log
                mean_value_loss += ppo_value_loss.item()
                mean_policy_loss += expert_action_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_policy_loss /= num_updates
        mean_value_loss /= num_updates
        return mean_policy_loss, mean_value_loss