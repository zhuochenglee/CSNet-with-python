from multiprocessing import freeze_support
import numpy
import torch
import torch.nn as nn
from att_cbam import CBAM
import cv2
import math
from att_se import eca_block


class ResidualBlock(nn.Module):
    def __init__(
        self, input_channels, num_channels, use_1x1conv=False, has_bn=False, strides=1
    ):
        super(ResidualBlock, self).__init__()
        self.has_bn = has_bn
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
        if self.has_bn:
            self.bn1 = nn.BatchNorm2d(input_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
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


def My_Reshape_Adap(input, blocksize):
    pass


# CSNet
class CSNet(nn.Module):
    def __init__(self, blocksize=32, subrate=0.1):
        super(CSNet, self).__init__()
        self.blocksize = blocksize
        self.samping = nn.Conv2d(
            1,
            int(numpy.round(blocksize * blocksize * subrate)),
            blocksize,
            stride=blocksize,
            padding=0,
            dilation=1,
            bias=False,
        )

        self.init_conv = nn.ConvTranspose2d(
            int(numpy.round(blocksize * blocksize * subrate)),
            1,
            32,
            stride=32,
            padding=0,
        )
        self.cbam = CBAM(64)
        self.se = eca_block(102)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        self.block2 = ResidualBlock(64, 64, has_bn=True)
        self.block3 = ResidualBlock(64, 64, has_bn=True)
        self.block4 = ResidualBlock(64, 64, has_bn=True)
        self.block5 = ResidualBlock(64, 64, has_bn=True)
        self.block6 = ResidualBlock(64, 64, has_bn=True)
        self.conv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        cs = self.samping(x)
        # 初始重建
        x = self.init_conv(cs)
        # 深度重建
        block1 = self.block1(x)  # out channels: 64
        block1_att = self.cbam(block1)
        block2 = self.block2(block1_att)
        block2_att = self.cbam(block2)
        block3 = self.block3(block2_att)
        block3_att = self.cbam(block3)
        block4 = self.block4(block3_att)
        block4_att = self.cbam(block4)
        block5 = self.block5(block4_att)
        block5_att = self.cbam(block5)
        block6 = self.block6(block5_att)
        block6_att = self.cbam(block6)
        block7 = self.conv(block6_att)
        block7_att = self.cbam(block7 + block1)
        res = x + block7_att
        return res


if __name__ == "__main__":
    freeze_support()
    img_path = "BSDS500/train/2018.jpg"
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to('cuda')
    tensor = tensor.float()
    net = CSNet()
    net = net.to('cuda')
    res = net(tensor)
    res = res.squeeze(0)
    res = res.squeeze(0)
    res = res.cpu().detach().numpy()
    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyWindow()
