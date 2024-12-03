
import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader

from model import AutoencoderPN
from loss import ChamferDistance
from dataset import ObjectTrajectoryDataset

from datetime import timedelta
from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs

# save a list of strings into txt
def save_list_strings(filename, data):
    with open(filename, "w") as file:
        for string in data:
            file.write(string + "\n")

# init weight
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        init.ones_(m.weight)
        init.zeros_(m.bias)

# train offline model bacth
def train_offline_model_batch(args):
    ## ------------------------ Init Accelerator ------------------------ ##
    # set up accelerator
    init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=40))
    accelerator = Accelerator(kwargs_handlers=[init_process_group_kwargs])

    ## ------------------------ Locate Folders ------------------------ ##
    # locate trajectory folder
    trajectory_name = 'results_trajectory_render'
    # /data0/v-wenbowang/Desktop/Logs/CVPR/full_train_best_0_static_init_body_grasp/results_trajectory_render
    trajectory_dir = osp.join('/data0/v-wenbowang/Desktop/Logs/CVPR/full_train_best_0_static_init_body_grasp/{}'.format(trajectory_name))
    
    # cluster mode: /mnt/blob/Desktop/
    if not osp.exists(trajectory_dir): 
        trajectory_dir = '/mnt/blob/Desktop/Logs/CVPR/full_train_best_0_static_init_body_grasp/{}'.format(trajectory_name)
    # locate log_dir
    log_dir = osp.join(trajectory_dir, 'Logs_{}_batch_{}'.format(args.mode, args.batch_size))
    os.makedirs(log_dir, exist_ok=True)
    
    # init PointNetModel
    num_feature, num_point = 64, 1024
    PointNetModel = AutoencoderPN(num_feature, num_point) # num_feature 64, num_point 1024
    PointNetModel = nn.SyncBatchNorm.convert_sync_batchnorm(PointNetModel)
    PointNetModel = PointNetModel.apply(weights_init)
    PointNetModel.train()
    # init PointNetModel Optimizer
    PointNetOptimizer = optim.Adam(PointNetModel.parameters(), lr=5e-4, weight_decay=1e-5)

    ## ------------------------ Load Trajectory Dataset ------------------------ ##
    num_object = args.finish - args.start + 1
    target_object_lines = list(range(args.start, args.finish+1))
    
    # init ObjectTrajectoryDatasetSimple
    OTDataset = ObjectTrajectoryDataset(trajectory_dir=trajectory_dir, target_object_lines=target_object_lines, 
                                        sample_ratio=1/args.batch_size, sample_object=True if args.mode=='Object' else False)
    # init ObjectTrajectoryDataLoader
    OTDataLoader = DataLoader(dataset=OTDataset, batch_size=args.batch_size, num_workers=16, shuffle=True, drop_last=True, pin_memory=True)

    ## ------------------------ Prepare Accelerator ------------------------ ##
    PointNetModel, PointNetOptimizer, OTDataLoader = accelerator.prepare(PointNetModel, PointNetOptimizer, OTDataLoader)

    ## ------------------------ Offline Train Actor Model ------------------------ ##
    # init L2 Loss Criterion
    Criterion = ChamferDistance()
    # init loss_logs
    loss_logs = {'losses': []}
    # init start_time and train_logs
    start_time, train_logs = time.time(), []
    train_epochs, group_size = 20 * args.batch_size, OTDataset.group_size
    num_iterations = OTDataLoader.__len__()

    # training epochs over entire dataset
    for nepoch in range(train_epochs + 1):
        # batch_traj: {'rendered': (Nbatch, Ngroup, 200, 2048, 4), 'valids': (Nbatch, Ngroup, 200, 1)}
        for niter, batch_traj in enumerate(OTDataLoader):
            # =============== Mini Batch =============== # #
            # generate permute indices
            permute_indices = torch.randperm(group_size)
            # =============== Unpack Batch =============== # #
            # train with mini_batch
            for mbatch in range(group_size):
                # generate mini_batch indices
                group_index = permute_indices[mbatch]

                # =============== Preprocess Batch =============== # #
                # get mini_batch trajectory data (Nbatch, 10, 200, 2048, 4), (Nbatch, 10, 200, 1)
                valids = batch_traj['valids'][:, group_index] # [Nbatch, 200, 1]
                rendered_pc = batch_traj['rendered'][:, group_index, :, :num_point, :3] #[Nbatch, 200, 1024, 3]

                # zero valid trajectory: learn first 10 frames
                if torch.sum(valids) == 0:
                    rendered_pc = rendered_pc[:, :10].reshape(-1, *rendered_pc.shape[-2:])
                else:
                    valids = valids.reshape(-1, valids.shape[-1])
                    rendered_pc = rendered_pc.reshape(-1, *rendered_pc.shape[-2:])[valids.squeeze(-1) == 1.]
                # assert batch sample
                if rendered_pc.shape[0] == 1: rendered_pc = rendered_pc.repeat(10, 1, 1)

                # =============== Train Batch =============== # #
                # Predict, use PointNetModel.module here
                rendered_pc = rendered_pc.permute(0, 2, 1)
                _, reconstructed_pc = PointNetModel(rendered_pc) # feat_emb, reconstructed_pc
                # Compute losses
                Loss = Criterion(reconstructed_pc, rendered_pc)
                # Loss = torch.mean((pred_actions - actions) ** 2)
                if accelerator.is_main_process: loss_logs['losses'].append(Loss.item())
                # Backward pass and optimization
                PointNetOptimizer.zero_grad()
                accelerator.backward(Loss)
                # Step optimizer
                PointNetOptimizer.step()

            # =============== Log iteration =============== # #
            if accelerator.is_main_process and niter % 1000 == 0:
                # compute mean mse losses
                mean_loss = np.mean(loss_logs['losses'])
                # compute processing time
                process_time = (time.time() - start_time) / 60
                total_time = (process_time / (niter + num_iterations * nepoch + 1)) * (num_iterations * (train_epochs + 1))
                train_logs.append('Training nepoch {:03d}, niter {:05d}/{}, losses {:.5f}; Processed time:{:05d}/{:05d} mins; Left time: {:.2f} hours'.format(
                    int(nepoch), int(niter), int(num_iterations), mean_loss, int(process_time), int(total_time), (total_time - process_time) / 60))
                print(train_logs[-1])
                # update log
                save_list_strings(osp.join(log_dir, 'train.log'), train_logs)
                # reset loss_logs
                loss_logs = {'losses': []}
        
        # =============== Save Epoch =============== # #
        # save trained actorcritic model: 2 or 5
        save_rate = 2
        if nepoch % save_rate == 0:
            # wait for everyone
            accelerator.wait_for_everyone()
            # unwrap and save model
            unwrapped_model = accelerator.unwrap_model(PointNetModel)
            accelerator.save(unwrapped_model.state_dict(), os.path.join(log_dir, 'model_{}.pt'.format(nepoch)))



if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script with argparse')
    parser.add_argument('--mode', type=str, default='Entire', help='Entire or Object')
    parser.add_argument('--start', type=int, default=0, help='Start Line')
    parser.add_argument('--finish', type=int, default=3199, help='Finish Line')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
    args = parser.parse_args()

    # train offline_model batch
    train_offline_model_batch(args=args)