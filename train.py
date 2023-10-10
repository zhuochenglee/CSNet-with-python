import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_util import TrainDataset
import network
import argparse
import os
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

current_time = datetime.now().date()
writer = SummaryWriter(log_dir=f'./runs/exp{current_time}')

parser = argparse.ArgumentParser()
parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
parser.add_argument('--block_size', default=32, type=int, help='CS block size')
parser.add_argument('--sub_rate', default=0.1, type=float, help='sampling sub rate')
parser.add_argument('--batchsize', default=64, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=100, type=int, help='number of round to be trained')
parser.add_argument('--load_epochs', default=0, type=int)
parser.add_argument('--lr', default=0.001, type=int, help='learning rate')
parser.add_argument('--step_size', default=5000, type=int, help='when to adjustment of learning rate')
parser.add_argument('--dataset', default='BSDS500/processed_images', type=str, help='dataset path')
parser.add_argument('--patience', default=7000, type=int, help='early stopping')
opt = parser.parse_args()
CROP_SIZE = opt.crop_size
BLOCK_SIZE = opt.block_size
NUM_EPOCHS = opt.num_epochs
LOAD_EPOCHS = opt.load_epochs
BATCH_SIZE = opt.batchsize
LR = opt.lr
SETP_SIZE = opt.step_size
DATASET = opt.dataset
PATIENCE = opt.patience

dataset = TrainDataset(DATASET, CROP_SIZE, BLOCK_SIZE)
train_dataloader = DataLoader(dataset, num_workers=0, batch_size=BATCH_SIZE, shuffle=True)

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
print(f'using blocksize:{BLOCK_SIZE} cropsize:{CROP_SIZE} epochs:{NUM_EPOCHS} batchsize:{BATCH_SIZE}')
loss_fn = nn.MSELoss()
loss_fn.to(device)

optimizer = torch.optim.Adam(net.parameters(), LR, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, SETP_SIZE, gamma=0.4)
best_pth = float('inf')
counter_it = 0

start_time = time.time()
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
        # fake_img_np = np.squeeze(fake_img_np, axis=1)
        # target_np = np.squeeze(target_np, axis=1)

        # structural_similarity = ssim(fake_img_np, target_np)
        # print(structural_similarity)
        g_loss = loss_fn(fake_img, target)
        g_loss.backward()
        optimizer.step()

        running_res['g_loss'] += g_loss.item() * batch_size
        writer.add_scalar(tag="loss/train", scalar_value=running_res['g_loss'])
        # running_res['ssim'] += structural_similarity

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
                           save_dir + '/A_BEST.pth')
            else:
                torch.save(net.state_dict(), save_dir + '/net_epoch_%d_%6f.pth' % (
                    epoch, current_loss))
                counter_it += 1
        if counter_it == PATIENCE:
            print(f'连续{counter_it}轮未下降loss，已停止训练')
            break
        scheduler.step()
    # ssim的最大值为 1.0
    # avg_ssim = running_res['ssim'] / batch_size
    # print(f'avg_ssim:{avg_ssim}')
total_time = time.time() - start_time
print(f'total time:{total_time}')
writer.close()
