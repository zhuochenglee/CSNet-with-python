import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data_util import TrainDataset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from network import CSNet
from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, train_bar, epoch):
        running_res = {'batch_size': 0, 'g_loss': 0, 'ssim': 0}
        loss_fn = nn.HuberLoss().to(self.gpu_id)
        FIRST = False
        current_time = datetime.now().date()
        if not os.path.exists('runs'):
            os.makedirs('runs')
        with open('exp_counter.txt', 'r') as file:
            line = file.readline()
        if FIRST:
            line = "0\n"
        line = line.rstrip('\n')
        line = int(line)
        writer = SummaryWriter(log_dir=f'./runs/exp{current_time}_实验名_{line}')
        line += 1
        line = str(line)
        line = line + '\n'
        with open('exp_counter.txt', 'w') as file:
            file.writelines(line)



        for data, target in train_bar:
            bs = data.size(0)
            if bs <= 0:
                continue
            running_res['batch_size'] += bs
            target = target.to(self.gpu_id)
            data = data.to(self.gpu_id)
            self.optimizer.zero_grad()
            fake_img = self.model(data).to(self.gpu_id)
            g_loss = loss_fn(fake_img, target)
            g_loss.backward()
            self.optimizer.step()
            running_res['g_loss'] += g_loss.item() * bs
            if self.gpu_id == 0:
                writer.add_scalar('loss_g', g_loss.item(), epoch)



    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        train_bar = tqdm(self.train_data)
        """for source, targets in train_bar:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)"""
        self._run_batch(train_bar, epoch)


def _save_checkpoint(self, epoch):
    ckp = self.model.module.state_dict()
    PATH = "checkpoint.pt"
    torch.save(ckp, PATH)
    print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


def train(self, max_epochs: int):
    for epoch in range(max_epochs):
        self._run_epoch(epoch)
        if self.gpu_id == 0 and epoch % self.save_every == 0:
            self._save_checkpoint(epoch)


def load_train_objs():
    train_set = TrainDataset('BSDS500/train', 96, 32)  # load your dataset
    model = CSNet()  # load your model
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', default=100, type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', default=5, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=64, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
