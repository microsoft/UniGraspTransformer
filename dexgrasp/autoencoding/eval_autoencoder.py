import numpy as np
import torch
import json
from torch.utils.data import DataLoader
from dataset import PointCloudsDex
import os
import numpy as np
from os.path import join as pjoin
from model import  AutoencoderPN
from loss import ChamferDistance
from tqdm import tqdm
DEVICE = torch.device('cuda:0')



def eval(dataset_path):
    feat_dim = 64 # 64 or 128
    ckpt_path = f'./utils/autoencoding_zl/ckpts/PN_{feat_dim}/023980.pth'
   
    test_set = PointCloudsDex(dataset_path, random_rotate=False)

    test_loader = DataLoader(
        dataset=test_set, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )

    model  = AutoencoderPN(k=feat_dim, num_points=1024)
    ckpt_path = os.path.join(ckpt_path)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)
    model.cuda()

    loss_func = ChamferDistance()

    feat_emb_tmp = []

    model.eval()
    running_error = 0.0
    for x in tqdm(test_loader,total = len(test_loader)):
        with torch.no_grad():
            x = x.to(DEVICE)
            feat_emb, x_restored = model(x) # feat_emb: [BS, 64, 1], restored_pc: [BS, 3, 1024] 
            # feat_emb_tmp.append(feat_emb)
            loss = loss_func(x, x_restored)
            running_error += loss.item()
    print(f'Avg loss: {running_error/len(test_set)}')
    # feat_emb_tmp = torch.concatenate(feat_emb_tmp)


if __name__=='__main__':
    dataset_path = '/home/v-leizhou/zl_dev/AutoEncoder/Assets/meshdatav3_pc_fps'
    eval(dataset_path)
