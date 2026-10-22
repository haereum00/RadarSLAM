#!/usr/bin/env python3

import os

import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R

from radarize.config import cfg, update_config


def align_2d(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    Rm = U @ np.diag([1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return Rm, md - Rm @ ms


def write_ply(path, pts, rgb):
    with open(path, "wb") as f:
        f.write(
            (
                "ply\nformat binary_little_endian 1.0\n"
                f"element vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            ).encode()
        )
        rec = np.zeros(len(pts), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = pts, rgb
        f.write(rec.tobytes())


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="experiment configure file name", default=None, type=str)
    parser.add_argument("--seq", help="sequence name, e.g. seq20", required=True)
    parser.add_argument("--poses", help="'gt' or a TUM trajectory file", default="gt")
    parser.add_argument("--out", help="output path without extension", default=None)
    parser.add_argument("--thresh", help="occupancy probability threshold", default=0.5, type=float)
    parser.add_argument("--voxel", help="voxel size in metres (horizontal)", default=0.1, type=float)
    parser.add_argument(
        "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = args()
    update_config(cfg, args)
    c3 = cfg["UNET3D"]
    ra = cfg["DATASET"]["RA"]

    d = np.load(os.path.join(cfg["DATASET"]["PATH"], args.seq + ".npz"))
    t, pose_gt = d["time"], d["pose_gt"]
    pred_dir = os.path.join(cfg["OUTPUT_DIR"], c3["MODEL"]["NAME"])
    prob = np.load(os.path.join(pred_dir, args.seq + ".npz"))["prob"]
    n = len(prob)
    tf = t[1 : n + 1]

    if args.poses == "gt":
        p = pose_gt[1 : n + 1]
        xy, z = p[:, :2], p[:, 2]
        yaw = R.from_quat(p[:, 3:]).as_euler("ZYX")[:, 0]
    else:
        a = np.loadtxt(args.poses)
        st, sxy = a[:, 0], a[:, 1:3]
        syaw = np.unwrap(R.from_quat(a[:, 4:8]).as_euler("ZYX")[:, 0])
        keep = (st >= t[0]) & (st <= t[-1])
        g = np.stack([np.interp(st[keep], t, pose_gt[:, k]) for k in range(2)], 1)
        Rm, tr = align_2d(sxy[keep], g)
        q = np.clip(tf, st[0], st[-1])
        xy = np.stack([np.interp(q, st, sxy[:, k]) for k in range(2)], 1) @ Rm.T + tr
        yaw = np.interp(q, st, syaw) + np.arctan2(Rm[1, 0], Rm[0, 0])
        z = np.full(n, np.median(pose_gt[:, 2]))

    edges = np.array(c3["HEIGHT_EDGES"])
    zc = (edges[:-1] + edges[1:]) / 2
    rng = np.linspace(0, ra["RR_MAX"], prob.shape[2])
    ang = np.deg2rad(np.linspace(ra["RA_MIN"], ra["RA_MAX"], prob.shape[3]))
    RR, AA = np.meshgrid(rng, ang, indexing="ij")
    lx, ly = RR * np.cos(AA), RR * np.sin(AA)

    occ = prob > args.thresh * 255
    pts = []
    for i in range(n):
        c, s = np.cos(yaw[i]), np.sin(yaw[i])
        for k in range(len(zc)):
            m = occ[i, k]
            x, y = lx[m], ly[m]
            pts.append(np.stack([xy[i, 0] + c * x - s * y, xy[i, 1] + s * x + c * y, np.full(len(x), z[i] + zc[k])], 1))
    pts = np.concatenate(pts)
    vox = np.array([args.voxel, args.voxel, edges[1] - edges[0]])
    pts = np.unique(np.round(pts / vox).astype(np.int64), axis=0) * vox

    ramp = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37], [220, 70, 30]], np.uint8)
    h = np.clip(np.searchsorted(edges, pts[:, 2] - np.median(z), side="right") - 1, 0, len(zc) - 1)
    rgb = ramp[h % len(ramp)]

    tag = "gt" if args.poses == "gt" else "traj"
    out = args.out or os.path.join(pred_dir, f"{args.seq}_map3d_{tag}")
    write_ply(out + ".ply", pts.astype(np.float32), rgb)
    np.savez_compressed(out + ".npz", points=pts.astype(np.float32), height_slice=h.astype(np.uint8))
    print(f"{len(pts)} points -> {out}.ply")
