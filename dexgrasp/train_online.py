# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ast import arg
import numpy as np
import random

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_sarl import *
from utils.process_marl import process_MultiAgentRL, get_AgentIndex


def train():
    print("Algorithm: ", args.algo)
    agent_index = get_AgentIndex(cfg)

    if args.algo in ["ppo", "dagger", "dagger_value"]:
        # parse env model: shadow_hand_grasp.py or shadow_hand_random_load_vision.py
        # task = eval(shadow_hand_grasp); env = VecTaskPython(task, rl_device)
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
        # obtain ppo, dagger module from process_ppo, process_dagger, ...
        sarl = eval('process_{}'.format(args.algo))(args, env, cfg_train, logdir)
        # set max_iterations
        iterations = cfg_train["learn"]["max_iterations"]
        if args.max_iterations > 0: iterations = args.max_iterations
        # ppo.run(); daggervalue.run()
        sarl.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])
    else:
        print("Unrecognized algorithm!")


if __name__ == '__main__':
    # set set_printoptions
    set_np_formatting()
    
    # init default args: task (shadow_hand_grasp), alog(ppo), num_envs, cfg_env, cfg_train
    args, train_flag = get_args()
    
    # start train or test process
    if train_flag:
        # load configs for cfg_env(shadow_hand_grasp.yaml), cfg_train(ppo/config.yaml)
        cfg, cfg_train, logdir = load_cfg(args)
        # gymutil.parse_arguments with args and cfg
        sim_params = parse_sim_params(args, cfg, cfg_train)
        # set system random seed: 0
        set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
        # run train() with specific task, algo, train/test mode
        train()