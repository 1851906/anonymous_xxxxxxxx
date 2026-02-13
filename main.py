import os
import random
import pickle
import argparse
from tqdm import tqdm

import warnings

import torch
import torch.backends.cudnn as cudnn

from data import RWAVSDataset

from models.model import S2A_NVAS
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log-dir', type=str, default="./logs/")
    parser.add_argument('--output-dir', type=str, default="room2_0.1/")
    parser.add_argument('--result-dir', type=str, default="./results/")
    parser.add_argument('--resume-path', type=str)
    parser.add_argument('--best-ckpt', type=str)
    # dataset
    parser.add_argument('--data-root', type=str, default="./data_RWAVS/2/")
    parser.add_argument('--room_number', type=str, default="1")
    parser.add_argument('--subset_ratio', type=float, default=1.0)

    # model
    parser.add_argument('--use_ori', action="store_true")
    parser.add_argument('--use_boundary_token', action="store_true")
    parser.add_argument('--boundary_r_scale', type=float, default="10.0")
    parser.add_argument('--boundary_k_numfreqs', type=int, default="6")
    # train
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--save-freq', type=int, default=10, help='Frequency (in epochs) at which to save the model')
    parser.add_argument('--device', type=str, default="cuda:0")
    # eval
    parser.add_argument('--eval', action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    configs = {
        "room_number": args.room_number,
        "use_ori": args.use_ori,
        "use_boundary_token": args.use_boundary_token,
        "boundary_r_scale": args.boundary_r_scale,
        "boundary_k_numfreqs": args.boundary_k_numfreqs,       ####
    }
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        cudnn.benchmark = True

    device = args.device


    print(f"Room number: {args.room_number}")
    print(f"Use orientation: {args.use_ori}")
    print(f"Use boundary anchors: {args.use_boundary_token}")

    train_dataset = RWAVSDataset(
        args.data_root, "train",
        subset_ratio=args.subset_ratio, subset_strategy='uniform'
    )
    val_dataset = RWAVSDataset(args.data_root, "val")

    # ===== NEW: point cloud from train dataset (same scene) =====
    pcd_xyz, pcd_rgb = train_dataset.get_point_cloud()
    pcd_xyz_256, pcd_rgb_256 = train_dataset.get_interior_anchors()
    boundary_xy_32, _ = train_dataset.get_boundary_anchors()

    model = S2A_NVAS(configs=configs, pcd_xyz=pcd_xyz_256, pcd_rgb=pcd_rgb_256, boundary_xy=boundary_xy_32)
    model.to(device)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
    if args.resume_path:
        checkpoint = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"load parameters from {args.resume_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_path:
        optimizer.load_state_dict(checkpoint["optimizer"])

    trainer = Trainer(
        args, model,
        criterion=None,
        optimizer=optimizer,
        log_dir=args.log_dir,
        last_epoch=checkpoint["epoch"] if args.resume_path else -1,
        last_iter=checkpoint["iter"] if args.resume_path else -1,
        device=device,
    )

    st_epoch = checkpoint["epoch"] + 1 if args.resume_path else 0
    ed_epoch = args.max_epoch
    t = tqdm(total=ed_epoch - st_epoch, desc="[EPOCH]")
    if args.resume_path:
        del checkpoint
    for epoch in range(st_epoch, ed_epoch):
        trainer.train(train_dataloader)
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == ed_epoch:
            trainer.save_ckpt()
        trainer.epoch += 1
        t.update()
    t.close()


if __name__ == '__main__':
    main()