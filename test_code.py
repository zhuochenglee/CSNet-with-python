from multiprocessing import freeze_support
import torch.nn.functional as F
import numpy
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, has_bn=False, strides=1):
        super(ResidualBlock, self).__init__()
        self.has_bn = has_bn
        self.conv1 = SeparableConv2d(input_channels, num_channels, kernel_size=3, padding=1)
        if self.has_bn:
            self.bn1 = nn.BatchNorm2d(input_channels)
        self.prelu = nn.PReLU()
        self.conv2 = SeparableConv2d(input_channels, num_channels, kernel_size=3, padding=1)
        if self.has_bn:
            self.bn2 = nn.BatchNorm2d(input_channels)

    def forward(self, x):
        y = self.conv1(x)
        if self.has_bn:
            y = self.bn1(y)
        y = self.prelu(y)
        y = self.conv2(y)
        if self.has_bn:
            y = self.bn2(y)
        return x + y


class RB2O(nn.Module):
    def __init__(self, cs, init_recon, blocksize=96, subrate=0.1):
        super(RB2O, self).__init__()
        self.cs = cs
        self.init_recon = init_recon
        self.samping = nn.Conv2d(1, int(numpy.round(blocksize * blocksize * subrate)), 3,
                                       padding=0)  # 对初始重构图像进行采样提取特征，得到观测值
        self.conv1 = nn.Conv2d(int(numpy.round(blocksize * blocksize * subrate)), blocksize * blocksize,
                                     blocksize, padding=0, stride=blocksize)  # 通过初始重建图像的观测值生成预测图像
        self.samping_2 = nn.Conv2d(1, int(numpy.round(blocksize * blocksize * subrate)), 3,
                                         padding=0)  # 得到预测图像的观测值
        self.init = nn.Conv2d(int(numpy.round(blocksize * blocksize * subrate)), blocksize * blocksize,
                                       blocksize, padding=0)  # 通过观测值残差重新生成一张图像

    def forward(self):
        sampe_cs = self.samping(self.init_recon)
        prediction_img = self.conv1(sampe_cs)
        prediction_img_cs = self.samping_2(prediction_img)
        residual_cs = self.cs - prediction_img_cs
        init_res = self.init(residual_cs)
        result = init_res + self.init_recon
        return result


# CSNet
class CSNet(nn.Module):
    def __init__(self, blocksize=96, subrate=0.1):
        super(CSNet, self).__init__()
        self.blocksize = blocksize
        self.samping = nn.Conv2d(1, int(numpy.round(blocksize * blocksize * subrate)), blocksize,
                                       stride=blocksize, padding=0,
                                       bias=False)
        self.deconv = nn.ConvTranspose2d(int(numpy.round(blocksize * blocksize * subrate)), blocksize * blocksize,
                                         3, padding=0, stride=blocksize)
        # self.init_conv = nn.Conv2d(int(numpy.round(blocksize * blocksize * subrate)),
        #                            blocksize * blocksize, 1, stride=1, padding=0)
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=blocksize * blocksize, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.ReLU(),
        )
        self.c2_d1 = nn.Sequential(
            nn.Conv2d(in_channels=blocksize * blocksize, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.ReLU(),
        )
        self.c2_d2 = nn.Sequential(
            nn.Conv2d(in_channels=numpy.round(blocksize * blocksize), out_channels=32, kernel_size=3, stride=1,
                      dilation=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 1),
            nn.ReLU(),
        )
        self.c3_d1 = nn.Sequential(
            nn.Conv2d(in_channels=blocksize * blocksize, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.ReLU(),
        )
        self.c3_d2 = nn.Sequential(
            nn.Conv2d(in_channels=blocksize * blocksize, out_channels=32, kernel_size=3, stride=1, dilation=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 1),
            nn.ReLU(),
        )
        self.c3_d3 = nn.Sequential(
            nn.Conv2d(in_channels=blocksize * blocksize, out_channels=32, kernel_size=3, stride=1, dilation=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 7, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 7, 1),
            nn.ReLU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.block2 = ResidualBlock(64, 64, has_bn=True)
        self.block3 = ResidualBlock(64, 64, has_bn=True)
        self.block4 = ResidualBlock(64, 64, has_bn=True)
        self.conv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        cs = self.samping(x)
        """cs = F.interpolate(cs, scale_factor=2, mode='bilinear', align_corners=False)
        cs = self.dilation_conv(cs)
        print(cs.shape)"""
        # 初始重建
        x = self.deconv(cs)
        feature = RB2O(cs=cs, init_recon=x)
        # 深度重建
        c1_out = self.c1(x)
        c2_d1_out = self.c2_d1(x)
        c2_d2_out = self.c2_d2(x)
        c3_d1_out = self.c3_d1(x)
        c3_d2_out = self.c3_d2(x)
        c3_d3_out = self.c3_d3(x)
        c2_d2_out = F.interpolate(c2_d2_out, size=(c2_d1_out.size(2), c2_d1_out.size(3)), mode='bilinear',
                                  align_corners=False)
        c2_out = torch.cat([c2_d1_out, c2_d2_out], dim=1)
        c3_d2_out = F.interpolate(c3_d2_out, size=(c3_d1_out.size(2), c3_d1_out.size(3)), mode='bilinear',
                                  align_corners=False)
        c3_d3_out = F.interpolate(c3_d3_out, size=(c3_d1_out.size(2), c3_d1_out.size(3)), mode='bilinear',
                                  align_corners=False)
        c3_out = torch.cat([c3_d1_out, c3_d2_out, c3_d3_out], dim=1)
        # 将三个特征提取路线的结果全部按照通道维度拼接
        msf = torch.cat([c1_out, c2_out, c3_out], dim=1)
        block1 = self.block1(msf)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        res = self.conv(block4)
        return res


if __name__ == '__main__':
    import torch

    freeze_support()
    img_path = 'BSDS500/train/2018.jpg'
    img = Image.open(img_path).convert('L')
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    in_tensor = preprocess(img)
    in_tensor = in_tensor.unsqueeze(0)
    in_tensor = in_tensor.to('cuda')
    net = CSNet()
    net = net.to('cuda')
    out = net(in_tensor)
    print(out.shape)
    out = out.squeeze(0)
    out = out.squeeze(0)
    # out = out.permute(1, 0)
    out = out.detach().cpu().numpy()
    plt.imshow(out)
    plt.show()
    print(out.shape)
