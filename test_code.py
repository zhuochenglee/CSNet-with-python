from multiprocessing import freeze_support
import torch.nn.functional as F
import numpy
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, has_bn=False, strides=1):
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


# reshape and concat
'''
def reshape_concat(img, blocksize):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.expand(1, -1, -1, -1)
    data = torch.clone(img_tensor.data)
    b_ = data.shape[0]  # batch-size
    c_ = data.shape[1]  # channels
    w_ = data.shape[2]  # width
    h_ = data.shape[3]  # height
    output = torch.zeros(b_, int(c_ / blocksize / blocksize),
                         int(w_ * blocksize), int(h_ * blocksize))
    for i in range(0, w_):
        for j in range(0, h_):
            data_temp = data[:, :, i, j]
            data_temp = data_temp.view(b_, int(c_ / blocksize / blocksize), blocksize, blocksize)
            output[:, :, i * blocksize: (i + 1) * blocksize, j * blocksize: (j + 1) * blocksize] += data_temp
    return output
'''


class Reshape_Concat_Adap(torch.autograd.Function):
    blocksize = 0

    def __init__(self, block_size):
        # super(Reshape_Concat_Adap, self).__init__()
        Reshape_Concat_Adap.blocksize = block_size

    @staticmethod
    def forward(ctx, input_, ):
        ctx.save_for_backward(input_)

        data = torch.clone(input_.data)
        b_ = data.shape[0]
        c_ = data.shape[1]
        w_ = data.shape[2]
        h_ = data.shape[3]

        output = torch.zeros((b_,
                              int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                              int(w_ * Reshape_Concat_Adap.blocksize),
                              int(h_ * Reshape_Concat_Adap.blocksize))).cuda()

        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = data[:, :, i, j]
                # data_temp = torch.zeros(data_t.shape).cuda() + data_t
                # data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                                            Reshape_Concat_Adap.blocksize, Reshape_Concat_Adap.blocksize))
                # print data_temp.shape
                output[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize] += data_temp

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        input_ = torch.clone(inp.data)
        grad_input = torch.clone(grad_output.data)

        b_ = input_.shape[0]
        c_ = input_.shape[1]
        w_ = input_.shape[2]
        h_ = input_.shape[3]

        output = torch.zeros((b_, c_, w_, h_)).cuda()
        output = output.view(b_, c_, w_, h_)
        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = grad_input[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                            j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize]
                # data_temp = torch.zeros(data_t.shape).cuda() + data_t
                data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, c_, 1, 1))
                output[:, :, i, j] += torch.squeeze(data_temp)

        return Variable(output)


def My_Reshape_Adap(input, blocksize):
    return Reshape_Concat_Adap(blocksize).apply(input)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "核大小必须为3或者7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out.values], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x


# CSNet
class CSNet(nn.Module):
    def __init__(self, blocksize=32, subrate=0.1):
        super(CSNet, self).__init__()
        self.blocksize = blocksize
        self.samping = nn.Conv2d(1, int(numpy.round(blocksize * blocksize * subrate)), blocksize,
                                 stride=blocksize, padding=18, dilation=2,
                                 bias=False)
        self.att = CBAM(int(numpy.round(blocksize * blocksize)), ratio=16, kernel_size=3)

        self.init_conv = nn.Conv2d(int(numpy.round(blocksize * blocksize * subrate)),
                                   blocksize * blocksize, 1, stride=1, padding=0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.block2 = ResidualBlock(64, 64, has_bn=True)
        self.block3 = ResidualBlock(64, 64, has_bn=True)
        self.block4 = ResidualBlock(64, 64, has_bn=True)
        self.block5 = ResidualBlock(64, 64, has_bn=True)
        self.block6 = ResidualBlock(64, 64, has_bn=True)
        self.conv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """cs = self.fc(x)
        cs = cs.clamp(0,1)
        cs = cs.squeeze(1)
        cs = torch.mean(cs, dim=0, keepdim=True)
        img = transforms.ToPILImage()(cs)
        plt.imshow(img)
        plt.show()"""
        cs = self.samping(x)
        """cs = F.interpolate(cs, scale_factor=2, mode='bilinear', align_corners=False)
        cs = self.dilation_conv(cs)
        print(cs.shape)"""
        # 初始重建

        x = self.init_conv(cs)
        x = self.att(x)
        x = My_Reshape_Adap(x, self.blocksize)
        # 深度重建
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.conv(block6)
        res = x + block7
        return res


if __name__ == '__main__':
    import torch

    freeze_support()
    img = torch.randn(1, 1, 180, 180)
    img = img.to('cuda')
    net = CSNet()
    net = net.to('cuda')
    out = net(img)
    print(out.detach().cpu().numpy())
    print(out.shape)
