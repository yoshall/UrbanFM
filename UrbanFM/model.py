import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math


class N2_Normalization(nn.Module):
    def __init__(self, upscale_factor):
        super(N2_Normalization, self).__init__()
        self.upscale_factor = upscale_factor
        self.avgpool = nn.AvgPool2d(upscale_factor)
        self.upsample = nn.Upsample(
            scale_factor=upscale_factor, mode='nearest')
        self.epsilon = 1e-5

    def forward(self, x):
        out = self.avgpool(x) * self.upscale_factor ** 2 # sum pooling
        out = self.upsample(out)
        return torch.div(x, out + self.epsilon)


class Recover_from_density(nn.Module):
    def __init__(self, upscale_factor):
        super(Recover_from_density, self).__init__()
        self.upscale_factor = upscale_factor
        self.upsample = nn.Upsample(
            scale_factor=upscale_factor, mode='nearest')

    def forward(self, x, lr_img):
        out = self.upsample(lr_img)
        return torch.mul(x, out)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(),
                      nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class UrbanFM(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=16,
                 base_channels=64, ext_dim=7, img_width=32, img_height=32, ext_flag=True, scaler_X=1500, scaler_Y=100):
        super(UrbanFM, self).__init__()
        self.ext_flag = ext_flag
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.img_width = img_width
        self.img_height = img_height

        if ext_flag:
            self.embed_day = nn.Embedding(8, 2) # Monday: 1, Sunday:7, ignore 0, thus use 8
            self.embed_hour = nn.Embedding(24, 3) # hour range [0, 23]
            self.embed_weather = nn.Embedding(18, 3) # ignore 0, thus use 18

            self.ext2lr = nn.Sequential(
                nn.Linear(12, 128),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Linear(128, img_width * img_height),
                nn.ReLU(inplace=True)
            )

            self.ext2hr = nn.Sequential(
                nn.Conv2d(1, 4, 3, 1, 1),
                nn.BatchNorm2d(4),
                nn.PixelShuffle(upscale_factor=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 4, 3, 1, 1),
                nn.BatchNorm2d(4),
                nn.PixelShuffle(upscale_factor=2),
                nn.ReLU(inplace=True),
            )

        if ext_flag:
            conv1_in = in_channels + 1
            conv3_in = base_channels + 1
        else:
            conv1_in = in_channels
            conv3_in = base_channels

        # input conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(conv1_in, base_channels, 9, 1, 4),
            nn.ReLU(inplace=True)
        )
        # output conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv3_in, out_channels, 9, 1, 4),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(base_channels))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1), nn.BatchNorm2d(base_channels))

        # Distributional upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
                           nn.BatchNorm2d(base_channels * 4),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

    def forward(self, x, ext):
        inp = x

        if self.ext_flag:
            ext_out1 = self.embed_day(ext[:, 4].long().view(-1, 1)).view(-1, 2)
            ext_out2 = self.embed_hour(
                ext[:, 5].long().view(-1, 1)).view(-1, 3)
            ext_out3 = self.embed_weather(
                ext[:, 6].long().view(-1, 1)).view(-1, 3)
            ext_out4 = ext[:, :4]
            ext_out = self.ext2lr(torch.cat(
                [ext_out1, ext_out2, ext_out3, ext_out4], dim=1)).view(-1, 1, self.img_width, self.img_height)
            inp = torch.cat([x, ext_out], dim=1)

        out1 = self.conv1(inp)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)

        # concatenation backward
        if self.ext_flag:
            ext_out = self.ext2hr(ext_out)
            out = self.conv3(torch.cat([out, ext_out], dim=1))
        else:
            out = self.conv3(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x * self.scaler_X / self.scaler_Y)
        return out
