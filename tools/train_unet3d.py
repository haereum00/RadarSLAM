#!/usr/bin/env python3

import os

import argparse
import random

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

torch.backends.cudnn.benchmark = True

from radarize.config import cfg, update_config
from radarize.unet import dataloader3d, model


def soft_dice_loss(y_pred, y, eps=1.0):
    inter = (y_pred * y).sum(dim=(0, 2, 3))
    denom = y_pred.sum(dim=(0, 2, 3)) + y.sum(dim=(0, 2, 3))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def slice_f1(y_pred, y):
    p = y_pred > 0.5
    t = y > 0.5
    tp = (p & t).sum(dim=(0, 2, 3)).double()
    fp = (p & ~t).sum(dim=(0, 2, 3)).double()
    fn = (~p & t).sum(dim=(0, 2, 3)).double()
    return tp, fp, fn


def train(net, device, train_loader, optimizer, epoch):
    net.train()
    loss_plot = 0
    for batch_idx, batch in enumerate(train_loader):
        for k, v in batch.items():
            batch[k] = v.to(device)
        x = dataloader3d.stack_input(batch)
        y = batch["depth_map_3d"]

        optimizer.zero_grad()
        y_pred = net(x)
        loss = cfg["UNET3D"]["TRAIN"]["BCE_WEIGHT"] * F.binary_cross_entropy(y_pred, y) + cfg["UNET3D"]["TRAIN"][
            "DICE_WEIGHT"
        ] * soft_dice_loss(y_pred, y)
        loss.backward()
        optimizer.step()
        loss_plot += loss.item()

        if batch_idx % cfg["UNET3D"]["TRAIN"]["LOG_STEP"] == 0:
            print(f"Train Epoch: {epoch} [({100. * batch_idx / len(train_loader):.0f}%)]\t Loss: {loss.item():.6f}")
    return loss_plot / len(train_loader)


def test(net, device, test_loader):
    net.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            y_pred = net(dataloader3d.stack_input(batch))
            a, b, c = slice_f1(y_pred, batch["depth_map_3d"])
            tp, fp, fn = tp + a, fp + b, fn + c
    f1 = (2 * tp / (2 * tp + fp + fn + 1e-9)).cpu().numpy()
    print(f"\n[Val] F1 per slice: {np.round(f1, 3)}  mean {f1.mean():.4f}")
    return 1 - f1.mean(), f1


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="experiment configure file name", default=None, type=str)
    parser.add_argument(
        "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER
    )
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = args()
    update_config(cfg, args)
    c3 = cfg["UNET3D"]

    train_dir = os.path.join(cfg["OUTPUT_DIR"], c3["MODEL"]["NAME"])
    os.makedirs(train_dir, exist_ok=True)

    random.seed(c3["TRAIN"]["SEED"])
    np.random.seed(c3["TRAIN"]["SEED"])
    torch.manual_seed(c3["TRAIN"]["SEED"])

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Loading dataset...")
    train_dataset = torch.utils.data.ConcatDataset(
        [
            dataloader3d.UNet3DDataset(os.path.join(cfg["DATASET"]["PATH"], x + ".npz"), transform=dataloader3d.FlipRange3D(0.5))
            for x in sorted(cfg["DATASET"]["TRAIN_SPLIT"])
        ]
    )
    val_dataset = torch.utils.data.ConcatDataset(
        [dataloader3d.UNet3DDataset(os.path.join(cfg["DATASET"]["PATH"], x + ".npz")) for x in sorted(cfg["DATASET"]["VAL_SPLIT"])]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=c3["TRAIN"]["BATCH_SIZE"], shuffle=True, drop_last=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=c3["TEST"]["BATCH_SIZE"], shuffle=False, num_workers=0)

    model_kwargs = {"n_channels": c3["MODEL"]["N_CHANNELS"], "n_classes": c3["MODEL"]["N_CLASSES"]}
    net = getattr(model, c3["MODEL"]["TYPE"])(**model_kwargs).to(device)
    optimizer = optim.Adam(net.parameters(), lr=c3["TRAIN"]["LR"], betas=(0.8, 0.9))

    train_losses, val_losses = [], []
    best = np.inf
    for epoch in range(1, c3["TRAIN"]["EPOCHS"] + 1):
        train_loss = train(net, device, train_loader, optimizer, epoch)
        val_loss, f1 = test(net, device, val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        plt.plot(train_losses, "b", label="Train loss (BCE + Dice)")
        plt.plot(val_losses, "r", label="Val 1 - mean F1")
        plt.title(f"UNet3D, best val 1-F1: {np.min(val_losses):.3f}")
        plt.legend()
        plt.grid()
        plt.ylim(bottom=0)
        plt.savefig(os.path.join(train_dir, "loss.jpg"))
        plt.close()

        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model_name": c3["MODEL"]["NAME"],
                    "model_type": type(net).__name__,
                    "model_state_dict": net.state_dict(),
                    "model_kwargs": model_kwargs,
                    "epoch": epoch,
                    "val_f1": f1.tolist(),
                    "height_edges": list(c3["HEIGHT_EDGES"]),
                },
                os.path.join(train_dir, f"{c3['MODEL']['NAME']}.pth"),
            )
