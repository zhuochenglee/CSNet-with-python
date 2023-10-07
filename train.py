import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_util import TrainDataset
from data_util import ssim
import network
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
parser.add_argument('--block_size', default=32, type=int, help='CS block size')
parser.add_argument('--sub_rate', default=0.1, type=float, help='sampling sub rate')
parser.add_argument('--batchsize', default=64, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--load_epochs', default=0, type=int)
opt = parser.parse_args()

CROP_SIZE = opt.crop_size
BLOCK_SIZE = opt.block_size
NUM_EPOCHS = opt.num_epochs
LOAD_EPOCHS = opt.load_epochs

# 在此处修改训练数据集路径
dataset = TrainDataset('BSDS500/train', CROP_SIZE, BLOCK_SIZE)

batchsize = 64
train_dataloader = DataLoader(dataset, num_workers=0, batch_size=batchsize, shuffle=True)

'''
for X in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    break
'''
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

net = network.CSNet(BLOCK_SIZE, opt.sub_rate).to(device)
print(net)

loss_fn = nn.MSELoss()
loss_fn.to(device)

optimizer = torch.optim.Adam(net.parameters(), 0.01, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.4)
best_pth = float('inf')

for epoch in range(LOAD_EPOCHS, NUM_EPOCHS + 1):
    train_bar = tqdm(train_dataloader)
    # print(train_bar)
    running_res = {'batch_size': 0, 'g_loss': 0, 'ssim': 0}
    net.train()
    # scheduler.step()
    # data和target是一模一样的图片
    for data, target in train_bar:
        # print(target)
        batch_size = data.size(0)
        if batch_size <= 0:
            continue
        running_res['batch_size'] += batch_size

        target = target.to(device)
        data = data.to(device)

        optimizer.zero_grad()
        fake_img = net(data).to(device)
        fake_img_np = fake_img.cpu().detach().numpy()
        target_np = target.cpu().detach().numpy()
        # print(numpy.asarray(fake_img_np.shape) - 7)\
        fake_img_np = np.squeeze(fake_img_np, axis=1)
        target_np = np.squeeze(target_np, axis=1)

        structural_similarity = ssim(fake_img_np, target_np)
        print(structural_similarity)
        g_loss = loss_fn(fake_img, target)
        g_loss.backward()
        optimizer.step()

        running_res['g_loss'] += g_loss.item() * batch_size
        running_res['ssim'] += structural_similarity

        train_bar.set_description(desc='[%d] Loss_G: %.7f lr: %.7f' % (
            epoch, running_res['g_loss'] / running_res['batch_size'], optimizer.param_groups[0]['lr']))
        save_dir = 'epochs' + '_subrate_' + str(opt.sub_rate) + '_blocksize_' + str(BLOCK_SIZE)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if epoch % 5 == 0:
            current_loss = running_res['g_loss'] / running_res['batch_size']
            if current_loss < best_pth:
                best_pth = current_loss
                torch.save(net.state_dict(),
                           save_dir + '/A_BEST')
            else:
                torch.save(net.state_dict(), save_dir + '/net_epoch_%d_%6f.pth' % (
                    epoch, current_loss))

        scheduler.step()
    # ssim的最大值为 1.0
    avg_ssim = running_res['ssim'] / batch_size
    print(f'avg_ssim:{avg_ssim}')