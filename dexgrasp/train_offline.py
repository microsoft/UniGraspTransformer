
import os
import tqdm
import time
import wandb
import argparse
import threading
import torch.nn as nn
import torch.optim as optim

from utils.general_utils import *
from torch.utils.data import DataLoader
from plot import plot_mlp_train_log_losses, plot_train_log_losses

from algorithms.rl.dagger_value.module import ActorCriticDagger
from algorithms.rl.dagger_value.module import ActorCriticTransformerEncoder, ActorCriticTransformerStepEncoder
from algorithms.rl.dagger_value.module import ActorCriticTransformerCausal, ActorCriticTransformerCausalEncoder

from algorithms.rl.dagger_value.dataset import ObjectTrajectoryDatasetBatch


# train offline model bacth
def train_offline_model_batch(args):
    # skip training for test seen and unseen set objects
    if args.object in ['test_set_seen_cat_results.yaml', 'test_set_unseen_cat_results.yaml']: return

    ## ------------------------ Locate Folders ------------------------ ##
    # set random seed
    set_seed(0, False)
    # locate and load config yaml
    config = load_yaml(osp.join(BASE_DIR, 'dexgrasp/cfg/train', args.config))
    # locate asset folder
    asset_dir = osp.join(BASE_DIR, '../Assets')
    # TODO: locate trajectory folder: use training
    trajectory_name = 'results_trajectory_train'
    if 'trajectory_name' in config['Offlines']: trajectory_name = config['Offlines']['trajectory_name']
    trajectory_dir = osp.join(BASE_DIR, '../Logs/{}/{}'.format(config['Infos']['save_name'], trajectory_name))
    # locate log_dir
    log_dir = osp.join(BASE_DIR, '../Logs/{}/results_distill/random/{}/distill_{:04d}_{:04d}_seed0'.format(config['Infos']['save_name'], config['Offlines']['save_name'], args.start, args.finish))
    
    # cluster mode: /mnt/blob/Desktop/
    if not osp.exists(asset_dir): 
        asset_dir = '/mnt/blob/Desktop/Assets'
        trajectory_dir = '/mnt/blob/Desktop/Logs/{}/{}'.format(config['Infos']['save_name'], trajectory_name)
        log_dir = osp.join('/mnt/blob/Desktop/Logs/{}/results_distill/random/{}/distill_{:04d}_{:04d}_seed0'.format(config['Infos']['save_name'], config['Offlines']['save_name'], args.start, args.finish))
    os.makedirs(log_dir, exist_ok=True)
    
    # check exsiting model
    if osp.exists(osp.join(log_dir, 'model_best.pt')):
        print('======== Find Existing Trained Model! ========')
        exit()

    # # ------------------------ Load Actor Critic Model ------------------------ ##
    # get model type: dagger_value, transformer, ...
    use_model_type = config['Distills']['use_model_type'] if 'use_model_type' in config['Distills'] else 'dagger_value'
    # adjust train_epochs and train_batchs for large number of objects
    num_object = args.finish - args.start + 1
    if num_object >= 3200: config['Offlines']['train_epochs'], config['Offlines']['train_batchs'] = 20, 200
    elif num_object >= 10: config['Offlines']['train_epochs'], config['Offlines']['train_batchs'] = 100, 100
    else: config['Offlines']['train_epochs'], config['Offlines']['train_batchs'] = 200, 100
    # sequential observations and actions input
    Sequential_Mode = True
    # use TransformerEcoder with observation patches and steps 
    if use_model_type == 'transformer_step_encoder':
        # init ActorCriticTransformerStepEncoder
        ActorCriticModel = ActorCriticTransformerStepEncoder(obs_shape=(config['Weights']['num_observation'], ), states_shape=None, actions_shape=(config['Weights']['num_action'], ), initial_std=0.8, model_cfg=config)
    # use TransformerEcoder with observation patches and sequences 
    elif use_model_type == 'transformer_causal_encoder':
        # init ActorCriticTransformerCausal
        ActorCriticModel = ActorCriticTransformerCausalEncoder(obs_shape=(config['Weights']['num_observation'], ), states_shape=None, actions_shape=(config['Weights']['num_action'], ), initial_std=0.8, model_cfg=config)
    # use TransformerEcoder with observation sequences
    elif use_model_type == 'transformer_causal':
        # init ActorCriticTransformerCausal
        ActorCriticModel = ActorCriticTransformerCausal(obs_shape=(config['Weights']['num_observation'], ), states_shape=None, actions_shape=(config['Weights']['num_action'], ), initial_std=0.8, model_cfg=config)
        # set training epochs and batchs
        if num_object >= 3200: config['Offlines']['train_epochs'], config['Offlines']['train_batchs'] = 40, 1000
        elif num_object >= 10: config['Offlines']['train_epochs'], config['Offlines']['train_batchs'] = 200, 500
        else: config['Offlines']['train_epochs'], config['Offlines']['train_batchs'] = 800, 500
    # use TransformerEcoder with observation patches 
    elif use_model_type == 'transformer_encoder':
        # use observation patches
        Sequential_Mode = False
        # init ActorCriticTransformerEncoder
        ActorCriticModel = ActorCriticTransformerEncoder(obs_shape=(config['Weights']['num_observation'], ), states_shape=None, actions_shape=(config['Weights']['num_action'], ), initial_std=0.8, model_cfg=config)
    # use MLP DaggerValue with observation entire
    elif use_model_type == 'dagger_value':
        # use observation patches
        Sequential_Mode = False
        # init ActorCriticDagger
        ActorCriticModel = ActorCriticDagger(obs_shape=(config['Weights']['num_observation'], ), states_shape=None, actions_shape=(config['Weights']['num_action'], ), initial_std=0.8, model_cfg=config['Offlines'])
        # modify learning rate
        if num_object >= 3200: config['Offlines']['learning_rate'] = float(config['Offlines']['learning_rate']) * 0.1 

    # send ActorCriticModel to GPU
    ActorCriticModel.to(args.device)
    # init ActorCriticModel Optimizer
    Optimizer_Actor = optim.AdamW(ActorCriticModel.actor.parameters(), lr=float(config['Offlines']['learning_rate']))
    # reduce train batchs with batch_ratio
    if 'batch_ratio' in config['Offlines']: config['Offlines']['train_batchs'] = int(config['Offlines']['train_batchs'] * config['Offlines']['batch_ratio'])

    # train state trajectory
    train_states = True if 'train_states' in config['Offlines'] and config['Offlines']['train_states'] else False

    ## ------------------------ Load Trajectory Dataset ------------------------ ##
    # locate object_scale_yaml
    object_scale_yaml = osp.join(BASE_DIR, 'results/state_based/{}'.format(args.object))
    # init ObjectTrajectoryDataset
    OTDataset = ObjectTrajectoryDatasetBatch(config=config, log_dir=log_dir, asset_dir=asset_dir, trajectory_dir=trajectory_dir, object_scale_yaml=object_scale_yaml, 
                                             target_object_lines=list(range(args.start, args.finish+1)), device=args.device)
    # init ObjectTrajectoryDataLoader
    OTDataLoader = DataLoader(dataset=OTDataset, batch_size=OTDataset.train_batchs, num_workers=8, shuffle=True, drop_last=True, pin_memory=True, worker_init_fn=lambda x: worker_init(x, 0))


    ## ------------------------ Offline Train Actor Model ------------------------ ##
    # init L2 Loss Criterion
    Criterion = nn.MSELoss()
    # init L1 Loss Criterion
    if 'l1_loss' in config['Offlines'] and config['Offlines']['l1_loss']: Criterion = nn.L1Loss()

    # create wandb
    if args.wandb: wandb.init(project='{}-{}'.format(log_dir.split('/')[-2], log_dir.split('/')[-1]))

    # init loss_logs
    loss_logs = {'losses': []}
    # init start_time and train_logs
    start_time, train_logs = time.time(), []
    batch_size, group_size = OTDataset.train_batchs, OTDataset.group_size
    current_process, total_process, log_process = 0, OTDataset.train_epochs * OTDataset.train_iterations, OTDataset.log_iterations
    # training epochs over entire dataset
    for nepoch in range(OTDataset.train_epochs + 1):
        # batch_traj: {'observations': (Nbatch, Ngroup, 200, Nobs), 'actions': (Nbatch, Ngroup, 200, Nact), 'valids': (Nbatch, Ngroup, 200, 1), 'successes': (Nbatch, Ngroup, 1)}
        for niter, batch_traj in enumerate(OTDataLoader):
            # =============== Mini Batch =============== # #
            # generate permute indices
            permute_indices = torch.randperm(group_size)

            # =============== Unpack Batch =============== # #
            # train with mini_batch
            for mbatch in range(group_size):
                # unpack mini_batch trajectory data
                group_index = permute_indices[mbatch]
                # get mini_batch trajectory data
                actions = batch_traj['actions'][:, group_index].to(args.device)
                observations = batch_traj['observations'][:, group_index].to(args.device)
                final_successes = batch_traj['final_successes'][:, group_index].to(args.device)
                valids = batch_traj['valids'][:, group_index].to(args.device)

                # train state trajectory
                if train_states:
                    # get batch_size, step_size
                    batch_size, step_size, _ = actions.shape
                    # get current_states (Nbatch, Nstep, Nstate)
                    current_states = batch_traj['states'][:, group_index].to(args.device)
                    current_states = current_states[:, :, :29]
                    # get next_states as prediction (Nbatch, Nstep, Nstate)
                    next_states = torch.cat([current_states[:, 1:, :], current_states[:, -1, :].unsqueeze(1)], dim=1)
                    # get fixed init_object_visual (Nbatch, Nstep, 64)
                    init_object_visual = observations[:, 0, config['Obs']['intervals']['object_visual'][0]:config['Obs']['intervals']['object_visual'][1]]
                    init_object_visual = init_object_visual.unsqueeze(1).repeat(1, step_size, 1)
                    # get current_times features (Nbatch, Nstep, 36)
                    current_times = compute_time_encoding(torch.arange(0, step_size), 36).to(current_states.device)
                    current_times = current_times.unsqueeze(0).repeat(batch_size, 1, 1)
                    
                    # pack new actions as next_states
                    actions = next_states
                    # pack new actions as deltas: next_states - current_states
                    if 'train_deltas' in config['Offlines'] and config['Offlines']['train_deltas']: actions = next_states - current_states
                    # pack new observations with current_states, init_object_visual, current_times
                    observations = torch.cat([current_states, init_object_visual, current_times], dim=-1)

                # keep trajectory sequence
                if Sequential_Mode:
                    # use final_successes trajectory
                    if torch.sum(final_successes) == 0: continue
                    actions = actions[final_successes.squeeze(-1) == 1.]
                    observations = observations[final_successes.squeeze(-1) == 1.]
                # patch trajectory sequence
                else:
                    # use valid trajectory patches
                    if 'use_valid_trajectory' in config['Offlines'] and config['Offlines']['use_valid_trajectory']:
                        if torch.sum(valids) == 0: continue
                        valids = valids.reshape(-1, valids.shape[-1])
                        actions = actions.reshape(-1, actions.shape[-1])[valids.squeeze(-1) == 1.]
                        observations = observations.reshape(-1, observations.shape[-1])[valids.squeeze(-1) == 1.]
                    # use final_successes trajectory
                    else:
                        if torch.sum(final_successes) == 0: continue
                        actions = actions[final_successes.squeeze(-1) == 1.].reshape(-1, actions.shape[-1])
                        observations = observations[final_successes.squeeze(-1) == 1.].reshape(-1, observations.shape[-1])

                # =============== Preprocess Batch =============== # #
                # Clip actions as in simulation
                if not train_states: actions = torch.clamp(actions, -1.0, 1.0)

                # =============== Train Batch =============== # #
                # Predict actions
                pred_actions, _ = ActorCriticModel.act_withgrad(observations)
                # Clip predictions as in simulation
                if not train_states: pred_actions = torch.clamp(pred_actions, -1.0, 1.0)
                # Compute losses
                Loss = Criterion(pred_actions, actions)
                # Loss = torch.mean((pred_actions - actions) ** 2)
                loss_logs['losses'].append(Loss.item())
                # Backward pass and optimization
                Optimizer_Actor.zero_grad()
                Loss.backward()
                # Step optimizer
                Optimizer_Actor.step()

                # =============== Log Batch =============== # #
                # update current_process
                current_process += 1
                if (niter * group_size + mbatch) % log_process == 0:
                    # compute mean mse losses
                    mean_loss = np.mean(loss_logs['losses'])
                    # compute processing time
                    process_time = (time.time() - start_time) / 60
                    total_time = (process_time / current_process) * total_process
                    train_logs.append('Training nepoch {:03d}, niter {:05d}/{:05d}, losses {:.5f}; Processed time:{:05d}/{:05d} mins; Left time: {:.2f} hours'.format(
                        int(nepoch), int(niter * group_size + mbatch), OTDataset.train_iterations, mean_loss, int(process_time), int(total_time), (total_time - process_time) / 60))
                    print(train_logs[-1])
                    # update log
                    if args.wandb: wandb.log({'Mean mse loss ': current_process})
                    save_list_strings(osp.join(log_dir, 'train.log'), train_logs)
                    # reset loss_logs
                    loss_logs = {'losses': []}

        # =============== Save Epoch =============== # #
        # save trained actorcritic model: 2 or 5
        save_rate = 5
        if nepoch % save_rate == 0: torch.save(ActorCriticModel.state_dict(), os.path.join(log_dir, 'model_{}.pt'.format(nepoch)))

        # save best trained mlp model
        if nepoch != 0 and nepoch % 10 == 0:
            # plot mlp training losses, find min loss and iteration
            min_loss, min_iter = plot_train_log_losses(log_dir, log_name='train.log', save_rate=save_rate)
            # update model_best.pt as min loss iteration model
            shutil.copy(osp.join(log_dir, 'model_{}.pt'.format(min_iter)), osp.join(log_dir, 'model_best.pt'))
            print('=============== Update Best Model ===============')
            print('Min_Loss {}: Min_Iter: {}'.format(min_loss, min_iter))
            print('=============== Update Best Model ===============')


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script with argparse')
    parser.add_argument('--start', type=int, default=None, help='Start Line')
    parser.add_argument('--finish', type=int, default=None, help='Finish Line')
    parser.add_argument('--config', type=str, default=None, help='Config File')
    parser.add_argument('--object', type=str, default=None, help='Object File')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device Number')
    parser.add_argument('--wandb', action="store_true", default=False, help='Create Wandb')
    args = parser.parse_args()

    # train offline_model batch
    train_offline_model_batch(args=args)