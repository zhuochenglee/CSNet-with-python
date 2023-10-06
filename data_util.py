import math
import numpy as np

import torch
from torch.utils.data import Dataset
import os
from os.path import join
from torchvision.transforms import (Compose, RandomResizedCrop, RandomHorizontalFlip,
                                    RandomVerticalFlip, RandomRotation, RandomCrop, ToTensor,
                                    ToPILImage, CenterCrop, Resize, Grayscale)
from PIL import Image
from skimage.metrics import structural_similarity


# 计算裁切比例
def calculate_crop_size(crop_size, block_size):
    return crop_size - (crop_size % block_size)


# 计算峰值信噪比
def psnr(pred, original, shave_border=0):
    height, width = pred[:2]
    height = len(height)
    width = len(width)
    # pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    original = Image.fromarray(original)
    original = original.resize((width, height))
    original = np.array(original)
    diff = pred - original
    rmse = math.sqrt(np.mean(diff ** 2))
    return 20 * math.log10(255.0 / rmse)


# 计算结构相似性 图像必须为灰度图
def ssim(img1, img2):
    return structural_similarity(img1, img2)


# 图像增强
def img_aug(crop_size):
    return Compose([
        RandomCrop(crop_size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        Grayscale(),
        ToTensor(),
    ])


class TrainDataset(Dataset):
    def __init__(self, img_file, crop_size, block_size):
        super(TrainDataset, self).__init__()
        self.img_names = [join(img_file, x) for x in os.listdir(img_file)]
        crop_size = calculate_crop_size(crop_size, block_size)
        self.transform = img_aug(crop_size)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        try:
            tr_image = self.transform(Image.open(self.img_names[item]))
            return tr_image, tr_image
        except:
            tr_image = self.transform(Image.open(self.img_names[item + 1]))
            return tr_image, tr_image


class TestimgDataset(Dataset):
    def __init__(self, img_file, blocksize):
        super(TestimgDataset, self).__init__()
        self.img_name = [join(img_file, x) for x in os.listdir(img_file)]
        self.blocksize = blocksize

    def __getitem__(self, item):
        hr_image = Image.open(self.img_name[item])
        w, h = hr_image.size
        w = int(np.float(w / self.blocksize) * self.blocksize)
        h = int(np.float(h / self.blocksize) * self.blocksize)
        cropsize = (h, w)
        hr_image = CenterCrop(cropsize)(hr_image)
        hr_image = Grayscale()(hr_image)
        return ToTensor()(hr_image), ToTensor()(hr_image)

    def __len__(self):
        return len(self.img_name)
