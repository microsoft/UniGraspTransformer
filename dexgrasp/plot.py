import os
import glob
import shutil
import argparse
import cv2 as cv
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from utils.general_utils import *


# plot train_log for offline training
def plot_train_log_losses(target_result_dir, log_name='train.log', save_rate=5):
    # init train_log_losses
    loss_dict = {'nepoch': [], 'losses': []}
    # locate and process loss.log
    log_file = osp.join(target_result_dir, log_name)
    # load log_file lines for niter_ngroup pair
    with open(log_file, 'r') as file: log_lines = file.readlines()

    # load data from log lines
    for log_line in log_lines:
        # locate loss_line
        loss_line = log_line.split(';')[0].split(',')
        # append nepoch
        loss_dict['nepoch'].append(int(loss_line[0].split(' ')[-1]))
        # append losses
        loss_dict['losses'].append(float(loss_line[-1].split(' ')[-1]))
    # load as array
    loss_dict['nepoch'] = np.asarray(loss_dict['nepoch'])
    loss_dict['losses'] = np.asarray(loss_dict['losses'])
    # get number of lines and epochs
    num_lines = len(loss_dict['losses'])
    num_epochs = max(loss_dict['nepoch']) + 1
    num_iters = np.count_nonzero(loss_dict['nepoch'] == 0)
    # append epoch losses
    epoch_losses = [np.mean(loss_dict['losses'][loss_dict['nepoch'] == nepoch]) for nepoch in range(num_epochs)]
    # sort loss_iters for saved models
    sorted_epoch_losses = np.argsort(epoch_losses)
    sorted_epoch_losses = [nepoch for nepoch in sorted_epoch_losses if nepoch % save_rate == 0]  # save rate: 2 or 5
    sorted_losses = [epoch_losses[nepoch] for nepoch in sorted_epoch_losses]
    # get min_loss_iter
    min_loss_epoch = sorted_epoch_losses[0]
    min_loss_line = min_loss_epoch * num_iters

    # plot losses over niter_ngroup_nsplit
    plt.figure(figsize=(15, 5))
    plt.semilogy(range(num_lines), loss_dict['losses'])
    plt.xticks(ticks=[n*num_iters for n in range(0, num_epochs + 1, 10)], labels=[n for n in range(0, num_epochs + 1, 10)])
    plt.axvline(x=min_loss_line+num_iters, color='red', linestyle='--', linewidth=2, label='Min Loss: {}'.format(['{:.4f}'.format(loss) for loss in sorted_losses[:5]]))
    plt.axvline(x=min_loss_line, color='red', linestyle='--', linewidth=2, label='Min Loss Iters: {}'.format(['{:04d}'.format(nepoch) for nepoch in sorted_epoch_losses[:5]]))
    plt.legend(loc='upper left', fontsize=12, title='Legend Title', title_fontsize='13')
    plt.xlabel('nepoch')
    plt.ylabel('losses')
    plt.title('Offline Training {}'.format(target_result_dir.split('/')[-1]))
    # save the loss figure
    save_name = log_file.split('/')[-1].split('.')[0]
    plt.savefig(osp.join(target_result_dir, '{}_losses_{}.png'.format(save_name, num_epochs-1)), format='png')
    # return min_loss and min_iter
    return sorted_losses[0], sorted_epoch_losses[0]


# plot single_object_train_results
def plot_single_object_train_results(target_result_dir, name, start=None, finish=None):
    # init success_dict
    success_dict, log_list = {'fail': [], 'okay': [], 'good': [], 'all': []}, []
    # process all result folders
    result_dirs = sorted(glob.glob(osp.join(target_result_dir, '*_seed0')))
    result_nums = len(result_dirs)
    if result_nums == 0: return [], 0, 0
    # process all results
    for nobj in range(result_nums):
        # locate result_dir
        result_dir = result_dirs[nobj]
        # locate result_id
        try: result_id = int(result_dirs[nobj].split('/')[-1].split('_')[0])
        except: continue
        # locate target start and end
        if start is not None and finish is not None and not (start <= result_id and result_id <= finish): continue
        # load and compute mean success_rate
        result_fn = osp.join(result_dir, '{}/final_success_rate.txt'.format(name))
        if not os.path.exists(result_fn): continue
        # compute mean success_rate
        success_rate = np.mean(np.loadtxt(result_fn))
        log_list.append('[{:04d}, {:.3f}],'.format(result_id, success_rate))
        print(log_list[-1])
        if success_rate < 0.6:
            success_dict['fail'].append([result_id, success_rate])
        elif success_rate < 0.9:
            success_dict['okay'].append([result_id, success_rate])
        else:
            success_dict['good'].append([result_id, success_rate])
        success_dict['all'].append([result_id, success_rate])
    
    # return with no result
    if len(success_dict['all']) == 0: return [], 0, 0
    
    # print final result
    for key in ['fail', 'okay', 'good', 'all']:
        log_list.append('{} Final Success: {} {} {}'.format(target_result_dir.split('/')[-1], key, len(success_dict[key]), '{:.3f}'.format(np.mean([success[1] for success in success_dict[key]])) if len(success_dict[key]) > 0 else 0))
        print(log_list[-1])
    print('Plot Result:', target_result_dir)
    
    # locate save name
    save_name = 'success' if start is None and finish is None else 'success_{}_{}'.format(start, finish)
    # save success_dict['all']
    save_list_strings(osp.join(target_result_dir, '{}.txt'.format(save_name)), log_list[-4:] + log_list)
    
    # Sort success_rate
    x = np.arange(len(success_dict['all']))
    sort_values = np.argsort(np.asarray(success_dict['all'])[:, 1])
    sort_values = np.asarray(success_dict['all'])[list(sort_values), :]
    # Create Figure with Bars
    plt.figure(figsize=(15, 5))
    plt.bar(x, sort_values[:, 1], width=1)
    # Labeling the axes
    plt.xlabel('Objects')
    plt.ylabel('Success Rate')
    # plt.title('Train from Scratch with Single Object')
    plt.title('Single Object Train-Test {}'.format(target_result_dir.split('/')[-1]))
    # Display the chart
    plt.savefig(osp.join(target_result_dir, '{}.png'.format(save_name)), format='png')
    # return log summary
    return log_list[-4:], int(log_list[-1].split(' ')[-2]), float(log_list[-1].split(' ')[-1])


# # plot all train results within results_train
# python plot.py --subfolder results_train --config train_best_0.yaml

# # plot all trajectory results within results_trajectory
# python plot.py --subfolder results_trajectory_test --name trajectory --config train_best_0.yaml

# # plot all train results within results_train trajectory_small
# python plot.py --subfolder results_trajectory --config train_best_0.yaml --name trajectory_small

# # plot all distill results within container
# python plot.py --subfolder results_distill/random --config transformer_encoder_concat_0.yaml


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script with argparse')
    parser.add_argument('--name', type=str, default='test_0', help='Test Name')
    parser.add_argument('--start', type=int, default=None, help='Start Line')
    parser.add_argument('--finish', type=int, default=None, help='Finish Line')
    parser.add_argument('--config', type=str, default=None, help='Config File')
    parser.add_argument('--subfolder', type=str, default=None, help='Subfolder Name')
    parser.add_argument('--container', action="store_true", default=False, help='Mount Container')
    args = parser.parse_args()

    # locate train config
    config_fn = osp.join(BASE_DIR, 'dexgrasp/cfg/train', args.config)
    if os.path.exists(config_fn):
        # load train config
        save_name = load_yaml(config_fn)['Infos']['save_name']
        # locate train result folder
        result_folders = [osp.join(BASE_DIR, '../Logs', save_name)]
        if not osp.exists(result_folders[0]): result_folders = [osp.join('/mnt/blob/Desktop/Logs', save_name)]
        # change log_dir to container
        if args.container: result_folders = [osp.join(BASE_DIR, '../Container/Desktop/Logs/', save_name)]
        # assign subfolder_dir
        if args.subfolder is not None: 
            if args.subfolder == 'results_distill/random' or args.subfolder == 'results_distill/group':
                result_folders = sorted(glob.glob(osp.join(result_folders[0], args.subfolder, load_yaml(config_fn)['Distills']['save_name'], '*_seed0')))
            else: result_folders = sorted(glob.glob(osp.join(result_folders[0], args.subfolder)))
        print('Plot Results:', len(result_folders), '\n', result_folders)
        # plot train results, save as success.txt
        result_logs, result_successes = [], []
        for result_folder in result_folders:
            print('Plot Result:', result_folder)
            if osp.exists(osp.join(result_folder, 'train.log')): plot_train_log_losses(result_folder, log_name='train.log')
            logs, nums, successes = plot_single_object_train_results(result_folder, args.name, args.start, args.finish)
            result_logs.append(logs)
            result_successes.append([nums, successes])
        # print all result_logs
        for result_log in result_logs:
            print('================ Summary ================')
            for log in result_log: print(log)
            print('=========================================')
        # print all successes    
        if len(result_successes) > 0:
            print('================ Successes ================')
            print('Result Successes', result_successes)
            result_successes = np.asarray(result_successes)
            print('Mean Successes:', np.sum(result_successes[:, 0] * result_successes[:, 1]) / max(1, np.sum(result_successes[:, 0])))
