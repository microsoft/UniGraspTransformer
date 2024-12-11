import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler


class RolloutStorage:

    def __init__(self, num_envs, buffer_size, obs_shape, states_shape, actions_shape, device='cpu', sampler='sequential'):
        # init hyper params
        self.device = device
        self.sampler = sampler
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.step = 0

        # Core
        self.observations = torch.zeros(buffer_size, num_envs, *obs_shape, device=self.device)
        self.rewards = torch.zeros(buffer_size, num_envs, 1, device=self.device)
        self.actions = torch.zeros(buffer_size, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(buffer_size, num_envs, 1, device=self.device).byte()
        # # TODO: record expert value
        self.values = None
        # self.values = torch.zeros(buffer_size, num_envs, 1, device=self.device)


    def add_transitions(self, observations, actions, rewards, dones, values=None):
        # locate current step
        self.step = self.step % self.buffer_size
        if self.step >= self.buffer_size: raise AssertionError("Rollout buffer overflow")
        # update buffer
        self.observations[self.step].copy_(observations)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        # # TODO: record expert value
        # self.values[self.step].copy_(values)

        self.step += 1

    def clear(self):
        self.step = 0

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.buffer_size
        mini_batch_size = batch_size // num_mini_batches

        if self.sampler == "sequential":
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch


class PPORolloutStorage:

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, states_shape, actions_shape, device='cpu', sampler='sequential'):
        # init hyper params
        self.device = device
        self.sampler = sampler
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.states = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # Expert
        self.actions_expert = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # # TODO: record expert value
        self.values_expert = None
        # self.values_expert = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        
        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)


    def add_transitions(self, observations, states, actions, rewards, dones, values, actions_log_prob, mu, sigma, actions_expert=None, values_expert=None):
        # locate current step
        self.step = self.step % self.num_transitions_per_env
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        # update buffer
        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)

        if actions_expert is not None: self.actions_expert[self.step].copy_(actions_expert)
        # # TODO: record expert value
        # if values_expert is not None: self.values_expert[self.step].copy_(values_expert)

        self.step += 1

    def clear(self):
        self.step = 0

    # compute PPO returns, advantages
    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        # values measure expected cumulative reward from current state following current policy
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # temporal difference error
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # advantages = sum(reward + next_value - current_value)
            # advantages measures how much better an action is compared to the average action
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # returns measures accumulated reward from current step
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        if self.sampler == "sequential":
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch


class PERBuffer:

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, states_shape, actions_shape, per_cfg, device='cpu'):
        # init hyper params
        self.device = device
        self.per_cfg = per_cfg
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.states = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        
        # For PER
        self.td_error = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)


    def add_transitions(self, observations, states, actions, rewards, dones, values, actions_log_prob, mu, sigma):
        # locate current step
        self.step = self.step % self.num_transitions_per_env
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
            
            # PER
            self.td_error[step] = delta

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
    
    def sample(self, batch_size):
        probs = (torch.abs(self.td_error.view(-1)) + self.per_cfg['eps']) ** self.per_cfg['alpha']
        probs /= probs.sum()
        total = len(probs)
        indices = np.random.choice(total, batch_size, p=probs.cpu().numpy(), replace=False)
        indices = torch.from_numpy(indices)
        weights = (total * probs[indices]) ** (-self.per_cfg['beta'])
        weights /= weights.max()
        
        return indices, weights

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        if self.sampler == "sequential":
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch
