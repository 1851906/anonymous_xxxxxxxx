import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
from losses import LogMagSTFTLoss

class Trainer(object):
    def __init__(self,
                 args,
                 model,
                 criterion,
                 optimizer,
                 log_dir,
                 last_epoch=-1,
                 last_iter=-1,
                 device='cuda',
                ):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        if log_dir:
            self.log_dir = os.path.join(log_dir, self.args.output_dir)
        self.epoch = last_epoch + 1
        self.max_epoch = self.args.max_epoch
        self.device = device
        self.iter_count = last_iter + 1
        if self.optimizer is not None:
            self.writer = SummaryWriter(self.log_dir)

        self.stft_loss = LogMagSTFTLoss(fft_size=512, shift_size=128, win_length=512,
                                        window="hamming_window")

    def train(self, train_loader):
        self.model.train()
        t = tqdm(total=len(train_loader), desc=f"[EPOCH {self.epoch} TRAIN]", leave=False)
        self.writer.add_scalar("epoch", self.epoch, self.epoch)
        for data in train_loader:
            for k in data.keys():
                data[k] = data[k].float().to(self.device)
            ret = self.model(data)
            pred_wav = ret["out_pred_wav"]
            gt_waveform = data["wav_bi"]
            min_len = min(pred_wav.shape[-1], gt_waveform.shape[-1])
            gt_waveform = gt_waveform[..., :min_len]
            pred_wav = pred_wav[..., :min_len]
            loss_bi = 20 * self.stft_loss(
                pred_wav.reshape(-1, pred_wav.shape[-1]).contiguous().float(),
                gt_waveform.reshape(-1, pred_wav.shape[-1]).contiguous().float()).mean()
            loss = loss_bi

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # adjust lr
            warmup_step = int(0.1 * self.args.max_epoch) * len(train_loader)
            if self.iter_count < warmup_step:
                lr = self.args.lr * self.iter_count / warmup_step
            else:
                lr = self.args.lr * 0.1 ** (2 * (self.iter_count - warmup_step) / (self.args.max_epoch * len(train_loader) - warmup_step))
            self.optimizer.param_groups[0]["lr"] = lr

            self.writer.add_scalar("train/lr", lr, self.iter_count)
            self.writer.add_scalar("train/loss_bi", loss_bi, self.iter_count)

            t.update()
            self.iter_count += 1
        t.close()




    def save_ckpt(self):
        try:
            state_dict = self.model.module.state_dict()  # remove prefix of multi GPUs
        except AttributeError:
            state_dict = self.model.state_dict()
        
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        torch.save({
                'epoch': self.epoch,
                'iter': self.iter_count,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict()},
                os.path.join(self.log_dir, f"{self.epoch}.pth"))