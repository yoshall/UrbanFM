import os
import sys
import warnings
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from .utils.metrics import get_MAE, get_MSE, get_MAPE
from .utils.data_process import get_dataloader, print_model_parm_nums

from .model import UrbanFM


warnings.filterwarnings("ignore")
# load arguments
parser = argparse.ArgumentParser()
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
parser.add_argument('--upscale_factor', type=int,
                    default=4, help='upscale factor')
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
model_path = 'saved_model/{}/{}-{}-{}'.format(opt.dataset,
                                              opt.n_residuals,
                                              opt.base_channels,
                                              opt.ext_flag)

# test CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# load model
model = UrbanFM(in_channels=opt.channels,
                out_channels=opt.channels,
                img_width=opt.img_width,
                img_height=opt.img_height,
                n_residual_blocks=opt.n_residuals,
                base_channels=opt.base_channels,
                ext_dim=opt.ext_dim,
                ext_flag=opt.ext_flag)
model.load_state_dict(torch.load('{}/final_model.pt'.format(model_path)))
model.eval()
if cuda:
    model.cuda()
print_model_parm_nums(model, 'UrbanFM')

# load test set
datapath = os.path.join('data', opt.dataset)
dataloader = get_dataloader(
    datapath, opt.scaler_X, opt.scaler_Y, 16, 'test')

total_mse, total_mae, total_mape = 0, 0, 0
for j, (test_data, ext, test_labels) in enumerate(dataloader):
    preds = model(test_data, ext).cpu().detach().numpy() * opt.scaler_Y
    test_labels = test_labels.cpu().detach().numpy() * opt.scaler_Y
    total_mse += get_MSE(preds, test_labels) * len(test_data)
    total_mae += get_MAE(preds, test_labels) * len(test_data)
    total_mape += get_MAPE(preds, test_labels) * len(test_data)
rmse = np.sqrt(total_mse / len(dataloader.dataset))
mae = total_mae / len(dataloader.dataset)
mape = total_mape / len(dataloader.dataset)

print('Test RMSE = {:.6f}, MAE = {:.6f}, MAPE = {:.6f}'.format(rmse, mae, mape))
