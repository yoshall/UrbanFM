import os
import sys
import warnings
import numpy as np
import argparse
import warnings
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .utils.metrics import get_MSE
from .utils.data_process import get_dataloader, print_model_parm_nums
from .model import UrbanFM, weights_init_normal

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100,
                    help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=16,
                    help='training batch size')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9,
                    help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999,
                    help='adam: decay of second order momentum of gradient')
parser.add_argument('--n_residuals', type=int, default=16,
                    help='number of residual units')
parser.add_argument('--base_channels', type=int,
                    default=128, help='number of feature maps')
parser.add_argument('--img_width', type=int, default=32,
                    help='image width')
parser.add_argument('--img_height', type=int, default=32,
                    help='image height')
parser.add_argument('--channels', type=int, default=1,
                    help='number of flow image channels')
parser.add_argument('--sample_interval', type=int, default=20,
                    help='interval between validation')
parser.add_argument('--harved_epoch', type=int, default=20,
                    help='halved at every x interval')
parser.add_argument('--upscale_factor', type=int,
                    default=4, help='upscale factor')
parser.add_argument('--seed', type=int, default=2017, help='random seed')
parser.add_argument('--scaler_X', type=int, default=1500,
                    help='scaler of coarse-grained flows')
parser.add_argument('--scaler_Y', type=int, default=100,
                    help='scaler of fine-grained flows')
parser.add_argument('--ext_dim', type=int, default=7,
                    help='external factor dimension')
parser.add_argument('--ext_flag', action='store_true',
                    help='whether to use external factors')
parser.add_argument('--dataset', type=str, default='P1',
                    help='which dataset to use')

opt = parser.parse_args()
print(opt)
torch.manual_seed(opt.seed)
warnings.filterwarnings('ignore')
# path for saving model
save_path = 'saved_model/{}/{}-{}-{}'.format(opt.dataset,
                                             opt.n_residuals,
                                             opt.base_channels,
                                             opt.ext_flag)
os.makedirs(save_path, exist_ok=True)


# test CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# initial model
model = UrbanFM(in_channels=opt.channels,
                out_channels=opt.channels,
                img_width=opt.img_width,
                img_height=opt.img_height,
                n_residual_blocks=opt.n_residuals,
                base_channels=opt.base_channels,
                ext_dim=opt.ext_dim,
                ext_flag=opt.ext_flag)
model.apply(weights_init_normal)
torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0)
print_model_parm_nums(model, 'UrbanFM')
criterion = nn.MSELoss()

if cuda:
    model.cuda()
    criterion.cuda()

# load training set and validation set
datapath = os.path.join('data', opt.dataset)
train_dataloader = get_dataloader(
    datapath, opt.scaler_X, opt.scaler_Y, opt.batch_size, 'train')
valid_dataloader = get_dataloader(
    datapath, opt.scaler_X, opt.scaler_Y, 4, 'valid')

# Optimizers
lr = opt.lr
optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, betas=(opt.b1, opt.b2))

# training phase
iter = 0
rmses = [np.inf]
maes = [np.inf]
for epoch in range(opt.n_epochs):
    train_loss = 0
    ep_time = datetime.now()
    for i, (flows_c, ext, flows_f) in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()

        # generate images with high resolution
        gen_hr = model(flows_c, ext)
        loss = criterion(gen_hr, flows_f)

        loss.backward()
        optimizer.step()

        print("[Epoch %d/%d] [Batch %d/%d] [Batch Loss: %f]" % (epoch,
                                                                opt.n_epochs,
                                                                i,
                                                                len(train_dataloader),
                                                                np.sqrt(loss.item())))

        # counting training mse
        train_loss += loss.item() * len(flows_c)

        iter += 1
        # validation phase
        if iter % opt.sample_interval == 0:
            model.eval()
            valid_time = datetime.now()
            total_mse, total_mae = 0, 0
            for j, (flows_c, ext, flows_f) in enumerate(valid_dataloader):
                preds = model(flows_c, ext)
                preds = preds.cpu().detach().numpy() * opt.scaler_Y
                flows_f = flows_f.cpu().detach().numpy() * opt.scaler_Y
                total_mse += get_MSE(preds, flows_f) * len(flows_c)
            rmse = np.sqrt(total_mse / len(valid_dataloader.dataset))
            if rmse < np.min(rmses):
                print("iter\t{}\tRMSE\t{:.6f}\ttime\t{}".format(iter, rmse, datetime.now()-valid_time))
                # save model at each iter
                # torch.save(UrbanFM.state_dict(),
                #            '{}/model-{}.pt'.format(save_path, iter))
                torch.save(model.state_dict(),
                           '{}/final_model.pt'.format(save_path))
                f = open('{}/results.txt'.format(save_path), 'a')
                f.write("epoch\t{}\titer\t{}\tRMSE\t{:.6f}\n".format(epoch, iter, rmse))
                f.close()
            rmses.append(rmse)

    # halve the learning rate
    if epoch % opt.harved_epoch == 0 and epoch != 0:
        lr /= 2
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(opt.b1, opt.b2))
        f = open('{}/results.txt'.format(save_path), 'a')
        f.write("half the learning rate!\n")
        f.close()

    print('=================time cost: {}==================='.format(
        datetime.now()-ep_time))
