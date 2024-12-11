import math
import torch
import numpy as np
import torch.nn as nn
from typing import List, Optional, Tuple
from torch.distributions import MultivariateNormal
from algo.pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNet
from algo.pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNetWithInstanceInfo


class PointNetBackbone(nn.Module):
    def __init__(
        self,
        pc_dim: int,
        feature_dim: int,
        pretrained_model_path: Optional[str] = None,
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.pc_dim = pc_dim
        self.feature_dim = feature_dim
        self.backbone = getPointNet({
                'input_feature_dim': self.pc_dim,
                'feat_dim': self.feature_dim
            })

        if pretrained_model_path is not None:
            print("Loading pretrained model from:", pretrained_model_path)
            state_dict = torch.load(pretrained_model_path, map_location="cpu")["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False,)
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)
            
    
    def forward(self, input_pc):
        others = {}
        return self.backbone(input_pc), others


class TransPointNetBackbone(nn.Module):
    def __init__(
        self,
        pc_dim: int = 6,
        feature_dim: int = 128,
        state_dim: int = 191 + 29,
        use_seg: bool = True,
    ):
        super().__init__()

        cfg = {}
        cfg["state_dim"] = 191 + 29
        cfg["feature_dim"] = feature_dim
        cfg["pc_dim"] = pc_dim
        cfg["output_dim"] = feature_dim
        if use_seg: cfg["mask_dim"] = 2
        else: cfg["mask_dim"] = 0
        self.transpn = getPointNetWithInstanceInfo(cfg)

    def forward(self, input_pc):
        others = {}
        input_pc["pc"] = torch.cat([input_pc["pc"], input_pc["mask"]], dim = -1)
        return self.transpn(input_pc), others


class Actor(nn.Module): # mlp actor model only

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False, use_pc=False):
        super(Actor, self).__init__()
        
        # load hyper params
        self.use_pc = use_pc
        self.asymmetric = asymmetric

        # load mlp model size
        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        # change observation space
        self.num_obs = obs_shape[0]
        # apply sigmoid-like layer nn.Tanh()
        self.sigmoid_actions = model_cfg['sigmoid_actions']
        
        # init actor layers
        actor_layers = []
        # create actor mlp layers
        actor_layers.append(nn.Linear(self.num_obs, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
                # apply sigmoid-like layer nn.Tanh()
                if self.sigmoid_actions: actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # # print actor and critic model
        # print("actor", self.actor)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        self.init_weights(self.actor, actor_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    # act without covariance but with gradient
    def act(self, observations):
        # forward actor model
        actions = self.actor(observations)
        return actions

    # act without covariance and gradient
    def act_inference(self, observations):
        # forward actor model
        actions = self.actor(observations)
        return actions.detach()


class ActorCriticDagger(nn.Module): # mlp actor and value models for dagger_value

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False, use_pc=False, combine=False):
        super(ActorCriticDagger, self).__init__()

        # load hyper params
        self.use_pc = use_pc
        self.combine = combine
        self.asymmetric = asymmetric

        # load mlp model size
        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']  # [1024, 1024, 512, 512]
            critic_hidden_dim = model_cfg['vf_hid_sizes']  # [1024, 1024, 512, 512]
            activation = get_activation(model_cfg['activation'])  # nn.ELU()
        # apply sigmoid-like layer nn.Tanh()
        self.sigmoid_actions = model_cfg['sigmoid_actions']
        
        # change observation space
        self.num_obs = obs_shape[0]

        # init actor and critic layers
        actor_layers = []
        critic_layers = []
        # create actor mlp layers
        actor_layers.append(nn.Linear(self.num_obs, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
                # apply sigmoid-like layer nn.Tanh()
                if self.sigmoid_actions: actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        # create critic mlp layers
        critic_layers.append(nn.Linear(self.num_obs, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # # print actor and critic models
        # print('actor', self.actor)
        # print('critic', self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    # act without gradient, with covariance
    def act(self, observations, states):
        # forward actor model
        actions_mean = self.actor(observations)
        # introduce action covariance
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # forward value model
        if self.asymmetric:
            value = self.critic(states)
        else:
            # combine value model with policy model for dagger_value
            if self.combine:
                value = self.critic(observations)
                return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()
            else:
                # no value gradient for vision backbone
                observations_copy = observations.clone()
                observations_copy.detach()
                value = self.critic(observations_copy)
        
        # note: dagger only use mean! should be more clear
        return actions_mean.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()
        #return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

    # act with gradient, without covariance
    def act_withgrad(self, observations, act_value=False):
        # forward actor model
        actions = self.actor(observations)
        # forward value model
        values = self.critic(observations) if act_value else None
        return actions, values

    # act without gradient, without covariance
    def act_inference(self, observations, act_value=False):
        # forward actor model
        actions_mean = self.actor(observations)
        # inference value model
        value_mean = self.critic(observations).detach() if act_value else None
        return actions_mean.detach(), value_mean

    # evaluate current actor model with previous collected actions
    def evaluate(self, observations, states, actions):
        # forward actor model
        actions_mean = self.actor(observations)
        # evaluate action covariance
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # evaluate value model
        if self.asymmetric:
            value = self.critic(states)
        else:
            # combine value model with policy model for dagger_value
            if self.combine:
                value = self.critic(observations)
            else:
                # no value gradient for vision backbone
                observations_copy = observations.clone()
                observations_copy.detach()
                value = self.critic(observations_copy)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


class ActorCritic(nn.Module): # mlp actor and value models for ppo, should be the same as ActorCritic(nn.Module) in ppo/module.py

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False, use_pc=False):
        super(ActorCritic, self).__init__()

        # load hyper params
        self.use_pc = use_pc
        self.asymmetric = asymmetric

        # load mlp model size
        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']  # [1024, 1024, 512, 512]
            critic_hidden_dim = model_cfg['vf_hid_sizes']  # [1024, 1024, 512, 512]
            activation = get_activation(model_cfg['activation'])  # nn.ELU()
        # apply sigmoid-like layer nn.Tanh()
        self.sigmoid_actions = model_cfg['sigmoid_actions']

        # change observation space
        self.num_obs = obs_shape[0]
        
        # init actor and critic layers
        actor_layers = []
        critic_layers = []
        # create actor mlp layers
        actor_layers.append(nn.Linear(self.num_obs, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
                # apply sigmoid-like layer nn.Tanh()
                if self.sigmoid_actions: actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        # create critic mlp layers
        critic_layers.append(nn.Linear(self.num_obs, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # # print actor and critic model
        # print('actor', self.actor)
        # print('critic', self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    # act without gradient, with covariance
    def act(self, observations, states):
        # forward actor model
        actions_mean = self.actor(observations)
        # introduce action covariance
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # forward value model
        if self.asymmetric: value = self.critic(states)
        else: value = self.critic(observations)

        return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

    # act without gradient, without covariance
    def act_inference(self, observations, act_value=False):
        # forward actor model
        actions_mean = self.actor(observations)
        # inference value also
        value_mean = self.critic(observations).detach() if act_value else None
        return actions_mean.detach(), value_mean

    # evaluate current actor model with previous collected actions
    def evaluate(self, observations, states, actions):
        # forward actor model
        actions_mean = self.actor(observations)
        # evaluate action covariance
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # evaluate value model
        if self.asymmetric: value = self.critic(states)
        else: value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


class SimpleMLP(nn.Module):  # one layer MLP
    def __init__(self, input_dim, output_dim, sigmoid_actions=False):
        super(SimpleMLP, self).__init__()
        # init mlp_layers
        mlp_layers = [nn.Linear(input_dim, output_dim)]
        # apply sigmoid-like layer nn.Tanh()
        if sigmoid_actions: mlp_layers.append(nn.Tanh())
        self.mlp_model = nn.Sequential(*mlp_layers)
    
    def forward(self, x):
        return self.mlp_model(x)


class LayerMLP(nn.Module):  # multi layers MLP
    def __init__(self, input_dim, output_dim, hidden_dims, activation, sigmoid_actions=False, init_weight=0.01):
        super(LayerMLP, self).__init__()
        # init mlp_layers and mlp_activation
        mlp_layers, mlp_activation = [], get_activation(activation)
        # create mlp_layers
        mlp_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        mlp_layers.append(mlp_activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                mlp_layers.append(nn.Linear(hidden_dims[l], output_dim))
                # apply sigmoid-like layer nn.Tanh()
                if sigmoid_actions: mlp_layers.append(nn.Tanh())
            else:
                mlp_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                mlp_layers.append(mlp_activation)
        self.mlp_model = nn.Sequential(*mlp_layers)

        # init the mlp_weights like in stable baselines
        mlp_weights = [np.sqrt(2)] * len(hidden_dims)
        mlp_weights.append(init_weight)
        self.init_weights(self.mlp_model, mlp_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self, x):
        return self.mlp_model(x)


class PositionalEncoding(nn.Module):  # positional encoding

    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()
        # init position and div_term
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # init positional embedding
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class ObsPatchTransformer(nn.Module):  # apply encode, self-attention, and decode on observation patches

    def __init__(self, obs_shape, actions_shape, model_cfg):
        super(ObsPatchTransformer, self).__init__()

        # load config
        self.config = model_cfg
        # load transformer encoder params
        self.num_heads = self.config['Offlines']['num_heads']
        self.num_layers = self.config['Offlines']['num_layers']
        self.num_features = self.config['Offlines']['num_features']
        # load observation patch infos
        self.obs_names = self.config['Obs']['names']
        self.obs_intervals = self.config['Obs']['intervals']
        # load observation and action shape
        self.num_obs = obs_shape[0]
        self.num_act = actions_shape[0]
        # load number of tokens, each with size num_features
        self.num_tokens = len(self.obs_names)

        # apply sigmoid-like layer nn.Tanh()
        self.sigmoid_actions = self.config["Offlines"]['sigmoid_actions']

        # # ---------------------- Create Encoder Layers ---------------------- # #
        # init mlp encoders for observation tokens
        self.mlp_encoders = nn.ModuleList([SimpleMLP(self.obs_intervals[name][1] - self.obs_intervals[name][0], self.num_features) for name in self.obs_names])
        
        # # ---------------------- Create Attention Layers ---------------------- # #
        # init positional encoder
        self.positional_encoder = PositionalEncoding(d_model=self.num_features, max_len=200)
        # init transformer self-attention encoders
        self.transformer_encoders = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_heads), num_layers=self.num_layers)
        
        # # ---------------------- Create Decoder Layers ---------------------- # #
        # init mlp decoder with concat_features
        if 'concat_features' in self.config['Offlines'] and self.config['Offlines']['concat_features']:
            # multi-layer MLP
            if 'decoder_hid_sizes' in self.config['Offlines']:
                self.decoder = LayerMLP(self.num_features * self.num_tokens, self.num_act, self.config['Offlines']['decoder_hid_sizes'],  self.config['Offlines']['decoder_activation'], self.sigmoid_actions)
            # one-layer MLP
            else: self.decoder = SimpleMLP(self.num_features * self.num_tokens, self.num_act, self.sigmoid_actions)  
        # init mlp decoder with average_features
        else: self.decoder = SimpleMLP(self.num_features, self.num_act, self.sigmoid_actions)

    # unpack observations (Nbatch, Nobs) into patches [(Nbatch, Npatch), ...]
    def unpack_observation_patches(self, observations):
        return [observations[..., self.obs_intervals[name][0]:self.obs_intervals[name][1]] for name in self.obs_names]
    
    # forward observations to actions
    def forward(self, observations):
        # # ---------------------- Forward Encoder Layers ---------------------- # #
        # get encoded tokens (Nbatch, Ntoken, Nfeat) from list of mlps and list of observation patches
        tokens = torch.stack([mlp_encoder(patch) for mlp_encoder, patch in zip(self.mlp_encoders, self.unpack_observation_patches(observations))], dim=-2)
        # introduce individual object visual feature
        if 'individual_object_visual_feature' in self.config['Offlines'] and self.config['Offlines']['individual_object_visual_feature']:
            object_visual_feature = torch.zeros((observations.shape[0], self.num_features), device=observations.device)
            object_visual_feature[:, :self.obs_intervals['object_visual'][1]-self.obs_intervals['object_visual'][0]] = observations[:, self.obs_intervals['object_visual'][0]:self.obs_intervals['object_visual'][1]]
            tokens[:, self.obs_names.index('object_visual'), :] = object_visual_feature

        # # ---------------------- Forward Transformer Layers ---------------------- # #
        # apply positional embedding
        tokens = self.positional_encoder(tokens)
        # get self-attention feats (Nbatch, Ntoken, Nfeat) from list of transformer encoders
        feats = self.transformer_encoders(tokens.transpose(0, 1)).transpose(0, 1)

        # # ---------------------- Create Decoder Layers ---------------------- # #
        # get outputs (Nbatch, Nact) from decoder with concat feats
        if 'concat_features' in self.config['Offlines'] and self.config['Offlines']['concat_features']:
            actions = self.decoder(feats.reshape(feats.shape[0], -1))
        # get outputs (Nbatch, Nact) from decoder with mean feats
        else: actions = self.decoder(torch.mean(feats, dim=-2))
        return actions


class ActorCriticTransformerEncoder(nn.Module):  # for transformer_encoder distillation with observation patches

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg):
        super(ActorCriticTransformerEncoder, self).__init__()

        # load config
        self.config = model_cfg
        # load observation action space
        self.num_obs = obs_shape[0]
        self.num_act = actions_shape[0]

        # init actor model as Transformer with observation patches
        self.actor = ObsPatchTransformer(obs_shape, actions_shape, model_cfg)
        # init critic model as MLP
        self.critic = LayerMLP(input_dim=self.num_obs, output_dim=1, hidden_dims=self.config['Offlines']['vf_hid_sizes'], 
                               activation=self.config['Offlines']['activation'], init_weight=1.0)

        # init action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

    def forward(self):
        raise NotImplementedError
    
    # act without gradient, with covariance
    def act(self, observations, states):
        # forward action model
        actions_mean = self.actor(observations)
        # introduce action covariance
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # no value gradient for vision backbone
        observations_copy = observations.clone()
        observations_copy.detach()
        value = self.critic(observations_copy)
        
        # note: dagger only use mean! should be more clear
        return actions_mean.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()
        #return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

    # act with gradient, without covariance
    def act_withgrad(self, observations, act_value=False):
        # forward actor model
        actions = self.actor(observations)
        # forward value model
        values = self.critic(observations) if act_value else None
        return actions, values

    # act without gradient, without covariance
    def act_inference(self, observations, act_value=False):
        # forward actor model
        actions_mean = self.actor(observations)
        # forward value model
        value_mean = self.critic(observations).detach() if act_value else None
        return actions_mean.detach(), value_mean

    # evaluate current actor model with previous collected actions
    def evaluate(self, observations, states, actions):
        # forward actor model
        actions_mean = self.actor(observations)
        # evaluate action covariance
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # no value gradient for vision backbone
        observations_copy = observations.clone()
        observations_copy.detach()
        value = self.critic(observations_copy)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)



def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None