import argparse
import numpy as np
import os
import time
import cv2
from PIL import Image
import torch
from torchvision.io import read_image
import torch.cuda
from torchvision.transforms import ToPILImage
from network import CSNet
import data_util

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--wab', default='epochs_subrate_0.1_blocksize_32/net_epoch_300_0.025189.pth',
                    type=str, help='weights and bais')
parser.add_argument('--test_data', default='testimg', type=str)
parser.add_argument('--block_size', default=32, type=int)
parser.add_argument('--sub_rate', default=0.1, type=float)
# parser.add_argument('save_dir', default="results", type=str)

opt = parser.parse_args()
DEVICE = opt.device
WAB = opt.wab
TEST_DATA = opt.test_data
BLOCK_SIZE = opt.block_size
SUB_RATE = opt.sub_rate
# SAVE_DIR = opt.save_dir

if DEVICE == 'cuda' and not torch.cuda.is_available():
    raise Exception("No GPU found")

model = CSNet(BLOCK_SIZE, SUB_RATE)
model.load_state_dict(torch.load(WAB))
model.eval()

img_list = []
for dirpath, dirnames, filenames in os.walk(TEST_DATA):
    for filename in filenames:
        img_list.append(os.path.join(dirpath, filename))
print('Found %d pictures' % len(img_list))
print(img_list)

avg_psnr_predicted = 0.0
avg_elapsed_time = 0.0

for img_file in img_list:
    print("processing ", img_file)
    img_ori_y = read_image(img_file)
    img_input = img_ori_y.float()
    # print(img_ori_y.dtype)
    img_input = img_input / 255.
    img_input = img_input.view(1, -1, img_ori_y.shape[1], img_ori_y.shape[2])
    # print(img_ori_y.shape)
    img_input = img_input.to(DEVICE)
    model = model.to(DEVICE)

    start_time = time.time()
    res = model(img_input)

    elapsed_time = time.time() - start_time

    avg_time = 0.0
    avg_time += elapsed_time

    img_tar_y = res.data[0].cpu().numpy().astype(np.float32)
    img_tar_y = img_tar_y * 255.
    img_tar_y[img_tar_y < 0] = 0
    img_tar_y[img_tar_y > 255.] = 255.

    save_img = img_tar_y[0, :, :]
    save_img = save_img.astype(np.uint8)
    # img_tar_y = img_tar_y.astype(np.uint8)
    # im = Image.fromarray(img_tar_y)
    # im.save("your_file.jpeg")

    SAVE_DIR = os.path.join('results', os.path.basename(TEST_DATA))
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    else:
        to_pil = ToPILImage()
        pil_img = to_pil(save_img)
        pil_img.save(os.path.join(SAVE_DIR, os.path.basename(img_file)))

    img_ori_y = img_ori_y.numpy().astype(np.uint8)
    img_ori_y = img_ori_y[0, :, :]
    img_tar_y = img_tar_y[0, :, :]
    img_tar_y = img_tar_y.astype(np.uint8)

    psnr_predicted = data_util.psnr(img_tar_y, img_ori_y, shave_border=0)
    print('PSNR on %s is %.4f' % (img_file, psnr_predicted))
    avg_psnr_predicted += psnr_predicted

print("dataset=", TEST_DATA)
print("PSNR_predicted=", avg_psnr_predicted / len(img_list))
print("average_time=", avg_time / len(img_list))
