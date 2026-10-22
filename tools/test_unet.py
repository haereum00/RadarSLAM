#!/usr/bin/env python3

import os

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

np.set_printoptions(precision=3, floatmode="fixed", sign=" ")

import matplotlib

matplotlib.use("Agg")
from collections import defaultdict


from radarize.config import cfg, update_config
from radarize.unet import model
from radarize.unet.dataloader import UNetDataset
from radarize.unet.dice_score import dice_loss


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", help="experiment configure file name", default=None, type=str
    )
    parser.add_argument("--npz_path", help="Path to npz.", default=None, required=False)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    return args


def test(net, device, test_loader):
    net.eval()

    test_dice_loss = 0
    test_mse_loss = 0

    runtimes = []

    d = defaultdict(list)
    m = {}

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            for k, v in batch.items():
                batch[k] = v.to(device)
            x = torch.cat(
                [
                    batch["radar_r_1"],
                    batch["radar_r_3"],
                    batch["radar_r_5"],
                    batch["radar_re_1"],
                    batch["radar_re_3"],
                    batch["radar_re_5"],
                ],
                dim=1,
            ).to(torch.float32)
            y = batch["depth_map"].to(torch.float32)

            tic = time.time()
            y_pred = net(x)
            y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
            toc = time.time()
            runtimes.append(toc - tic)

            d["time"].append(batch["time"][:, -1].cpu().numpy())
            d["radar_r"].append(batch["radar_r_3"].cpu().numpy())
            d["depth_map_pred"].append(y_pred.cpu().numpy().astype(np.float64))
            d["depth_map"].append(y.cpu().numpy().astype(np.float64))

            test_dice_loss += dice_loss(y_pred, y, multiclass=True).item()
            test_mse_loss += F.mse_loss(y_pred, y).item()

    test_dice_loss = test_dice_loss / len(test_loader)
    test_mse_loss = test_mse_loss / len(test_loader)
    print(f"\n[Test] Dice loss: {test_dice_loss:.6f} MSE loss: {test_mse_loss:.6f}")
    average_runtime = np.mean(runtimes)
    print(f"Average runtime: {average_runtime:.3f} seconds")

    for k, v in d.items():
        d[k] = np.concatenate(v, axis=0)
        print(k, d[k].shape, d[k].dtype)

    m["runtime"] = average_runtime
    m["dice_loss"] = test_dice_loss
    m["mse_loss"] = test_mse_loss

    return d, m


if __name__ == "__main__":
    args = args()
    update_config(cfg, args)

    device = torch.device("cpu" if args.no_cuda else "cuda")

    saved_model = torch.load(
        os.path.join(
            cfg["OUTPUT_DIR"],
            cfg["UNET"]["MODEL"]["NAME"],
            f"{cfg['UNET']['MODEL']['NAME']}.pth",
        )
    )
    model_name = saved_model["model_name"]
    model_type = saved_model["model_type"]
    state_dict = saved_model["model_state_dict"]
    model_kwargs = saved_model["model_kwargs"]
    net = getattr(model, model_type)(**model_kwargs).to(device)
    net.load_state_dict(state_dict)
    print(model_name, device)

    test_res_dir = os.path.join(os.path.join(cfg["OUTPUT_DIR"], model_name))
    if not os.path.exists(test_res_dir):
        os.makedirs(test_res_dir)

    if args.npz_path:
        npz_paths = [args.npz_path]
    else:
        npz_paths = sorted(
            [
                os.path.join(cfg["DATASET"]["PATH"], x + ".npz")
                for x in cfg["DATASET"]["TEST_SPLIT"]
            ]
        )

    for path in npz_paths:
        print(f"Processing {path}...")

        dataset = UNetDataset(path, seq_len=1)
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )
        d, m = test(net, device, test_loader)

        np.savez(
            os.path.join(test_res_dir, os.path.basename(path)),
            time=d["time"],
            depth_map=d["depth_map_pred"],
        )

        fname = os.path.join(
            test_res_dir, os.path.basename(path).replace(".npz", ".txt")
        )
        with open(fname, "w") as f:
            for k, v in m.items():
                f.write(f"{k}: {v:.6f}\n")
