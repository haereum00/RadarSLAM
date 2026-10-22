#!/usr/bin/env python3

import os

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from radarize.config import cfg, update_config
from radarize.unet import dataloader3d, model


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="experiment configure file name", default=None, type=str)
    parser.add_argument(
        "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER
    )
    parser.add_argument("--npz_path", help="Path to npz.", default=None, required=False)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def scores(pred, gt, tol=0):
    if tol:
        k = 2 * tol + 1
        gt_d = F.max_pool2d(gt, k, 1, tol)
        pred_d = F.max_pool2d(pred, k, 1, tol)
    else:
        gt_d, pred_d = gt, pred
    tp_p = (pred * gt_d).sum(dim=(0, 2, 3))
    tp_r = (gt * pred_d).sum(dim=(0, 2, 3))
    prec = tp_p / pred.sum(dim=(0, 2, 3)).clamp(min=1)
    rec = tp_r / gt.sum(dim=(0, 2, 3)).clamp(min=1)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
    return prec.cpu().numpy(), rec.cpu().numpy(), f1.cpu().numpy()


if __name__ == "__main__":
    args = args()
    update_config(cfg, args)
    c3 = cfg["UNET3D"]
    device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")

    out_dir = os.path.join(cfg["OUTPUT_DIR"], c3["MODEL"]["NAME"])
    ckpt = torch.load(os.path.join(out_dir, f"{c3['MODEL']['NAME']}.pth"), map_location=device)
    net = getattr(model, ckpt["model_type"])(**ckpt["model_kwargs"]).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    paths = [args.npz_path] if args.npz_path else [
        os.path.join(cfg["DATASET"]["PATH"], x + ".npz") for x in cfg["DATASET"]["TEST_SPLIT"]
    ]
    for path in paths:
        print(f"Processing {path}...")
        ds = dataloader3d.UNet3DDataset(path)
        loader = torch.utils.data.DataLoader(ds, batch_size=c3["TEST"]["BATCH_SIZE"], shuffle=False)
        preds, gts, times = [], [], []
        with torch.no_grad():
            for batch in tqdm(loader):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                preds.append(net(dataloader3d.stack_input(batch)).cpu())
                gts.append(batch["depth_map_3d"].cpu())
                times.append(batch["time"][:, -1].cpu())
        P = torch.cat(preds)
        G = torch.cat(gts)
        Pb = (P > 0.5).float()

        m = {"height_edges": list(c3["HEIGHT_EDGES"])}
        for name, tol in [("exact", 0), ("tol1", 1)]:
            pr, rc, f1 = scores(Pb, G, tol)
            m[name] = {"precision": pr.round(4).tolist(), "recall": rc.round(4).tolist(), "f1": f1.round(4).tolist(),
                       "mean_f1": float(f1.mean())}
        print(json.dumps(m, indent=1))

        base = os.path.join(out_dir, os.path.basename(path).replace(".npz", ""))
        np.savez_compressed(base + ".npz", time=np.concatenate(times), prob=(P.numpy() * 255).astype(np.uint8))
        with open(base + ".json", "w") as f:
            json.dump(m, f, indent=1)
