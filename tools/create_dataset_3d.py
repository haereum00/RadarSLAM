#!/usr/bin/env python3

import os
import sys

import argparse

import numpy as np
import rosbag
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import create_dataset as cd

from radarize.config import cfg, update_config

TX_SLOT = np.array([0] * 4 + [2] * 4 + [1] * 4)


def create_radar_elevation(
    bag,
    radar_params,
    radar_topic,
    angle_range,
    angle_bins,
    phase_bias,
    az_sign,
    el_sign,
    radar_buffer_len=3,
):
    n_samples = radar_params["n_samples"]
    win = np.hanning(n_samples)
    theta = az_sign * np.deg2rad(np.linspace(-angle_range, angle_range, angle_bins))
    steer = np.exp(1j * np.pi * np.outer(np.sin(theta), np.arange(2, 6)))

    buf_cross, buf_pow = [], []
    el_ts, el_msgs = [], []

    for topic, msg, ts in tqdm(
        bag.read_messages([radar_topic]), total=bag.get_message_count(radar_topic)
    ):
        cube = cd.read_radar_cube(msg)
        rf = np.fft.fft(cube * win, axis=2)

        z = np.sum(rf[1:, :4, :] * np.conj(rf[:-1, :4, :]), axis=(0, 1))
        rf = rf * np.exp(-1j * np.outer(TX_SLOT, np.angle(z)) / 3)[None]

        xa = np.einsum("ac,lcr->lar", np.conj(steer), rf[:, 2:6, :])
        xe = np.einsum("ac,lcr->lar", np.conj(steer), rf[:, 8:12, :])
        cross = np.sum(xe * np.conj(xa), axis=0).T
        power = np.mean(np.abs(xa) ** 2 + np.abs(xe) ** 2, axis=0).T

        buf_cross.append(cross)
        buf_pow.append(power)
        if len(buf_cross) > radar_buffer_len:
            buf_cross.pop(0)
            buf_pow.pop(0)
        if len(buf_cross) < radar_buffer_len:
            continue

        c = np.sum(buf_cross, axis=0) * np.exp(-1j * phase_bias)
        el_phase = el_sign * np.angle(c)
        lp = np.log(np.sum(buf_pow, axis=0) + 1e-9)
        lo, hi = np.percentile(lp, [5, 99.5])
        w = np.clip((lp - lo) / (hi - lo + 1e-9), 0, 1)

        el_msgs.append(np.stack([w * np.cos(el_phase), w * np.sin(el_phase)]).astype(np.float32))
        el_ts.append(ts.to_sec())

    return np.array(el_ts), np.array(el_msgs)


def create_lidar_3d(
    bag,
    radar_params,
    R_RL,
    t_RL,
    pose_ts,
    pose_msgs,
    lidar_topic,
    angle_range,
    angle_bins,
    height_edges,
    window=15.0,
    voxel_size=0.05,
    time_bucket=2.0,
    min_range=0.2,
    max_lidar_range=15.0,
):
    range_max = radar_params["range_max"]
    range_bins = radar_params["n_samples"]
    T_RL = cd.make_transform(R_RL, t_RL)
    rot = Slerp(pose_ts, R.from_quat(pose_msgs[:, 3:]))

    def pose_at(t):
        t = np.clip(t, pose_ts[0], pose_ts[-1])
        return rot(t), np.stack([np.interp(t, pose_ts, pose_msgs[:, k]) for k in range(3)], 1)

    z_lo, z_hi = height_edges[0], height_edges[-1]
    map_points, map_ts = [], []
    for topic, msg, ts in tqdm(
        bag.read_messages([lidar_topic]), total=bag.get_message_count(lidar_topic)
    ):
        pcd, pcd_ts = cd.livox_msg_to_numpy(msg)
        dist = np.linalg.norm(pcd, axis=1)
        keep = (dist > 0.3) & (dist < max_lidar_range)
        pcd, pcd_ts = pcd[keep], pcd_ts[keep]
        if len(pcd) == 0:
            continue

        rots, pos = pose_at(pcd_ts)
        pcd = rots.apply(pcd @ T_RL[:3, :3].T + T_RL[:3, 3]) + pos

        keep = (pcd[:, 2] - pos[:, 2] > z_lo - 0.2) & (pcd[:, 2] - pos[:, 2] < z_hi + 0.2)
        pcd, pcd_ts = pcd[keep], pcd_ts[keep]
        keys = np.hstack(
            [np.floor(pcd / voxel_size).astype(np.int64), np.floor(pcd_ts / time_bucket).astype(np.int64)[:, None]]
        )
        _, idxs = np.unique(keys, axis=0, return_index=True)
        map_points.append(pcd[idxs])
        map_ts.append(pcd_ts[idxs])

    map_points = np.concatenate(map_points)
    map_ts = np.concatenate(map_ts)
    tree = cKDTree(map_points[:, :2])
    print(f"LiDAR 3D map: {len(map_points)} points")

    range_axis = np.linspace(0, range_max, range_bins)
    angle_axis = np.linspace(np.deg2rad(-angle_range), np.deg2rad(angle_range), angle_bins)
    yaw = R.from_quat(pose_msgs[:, 3:]).as_euler("ZYX")[:, 0]

    maps = []
    for i in tqdm(range(len(pose_ts))):
        p = pose_msgs[i, :3]
        idxs = np.array(tree.query_ball_point(p[:2], range_max), dtype=np.int64)
        idxs = idxs[np.abs(map_ts[idxs] - pose_ts[i]) <= window]
        pts = map_points[idxs] - p
        c, s = np.cos(yaw[i]), np.sin(yaw[i])
        x = c * pts[:, 0] + s * pts[:, 1]
        y = -s * pts[:, 0] + c * pts[:, 1]
        z = pts[:, 2]
        keep = np.hypot(x, y) > min_range

        slices = []
        for k in range(len(height_edges) - 1):
            sel = keep & (z >= height_edges[k]) & (z < height_edges[k + 1])
            occ = cd.polar_occupancy_map(np.stack([-y[sel], x[sel]], 1), range_axis, angle_axis)
            slices.append(occ == 0)
        maps.append(np.stack(slices).astype(np.uint8))

    return np.array(maps)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="experiment configure file name", default=None, type=str)
    parser.add_argument(
        "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER
    )
    parser.add_argument("--bag_path", help="Path to bag.", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = args()
    update_config(cfg, args)
    c3 = cfg["UNET3D"]
    cfg = cfg["DATASET"]
    if cfg["RADAR_CHANNEL_ORDER"]:
        cd.RADAR_CHANNEL_ORDER = list(cfg["RADAR_CHANNEL_ORDER"])

    npz_path = os.path.splitext(args.bag_path)[0] + ".npz"
    print(f"Processing {args.bag_path} (frames from {npz_path})...")
    with np.load(npz_path) as d:
        times = d["time"]
        pose_gt = d["pose_gt"]

    with open(cfg["RADAR_CONFIG"], "r") as f:
        radar_params = cd.radar_config.RadarConfig(f.readlines()).get_params()
    R_RL, t_RL, _ = cd.load_lidar_radar_extrinsic(cfg["LIDAR_RADAR_EXTRINSICS"])
    bag = rosbag.Bag(args.bag_path)

    el_ts, el_msgs = create_radar_elevation(
        bag,
        radar_params,
        cfg["RADAR_TOPIC"],
        cfg["RA"]["RA_MAX"],
        cfg["RA"]["RAMAP_ASIZE"],
        np.deg2rad(c3["ELEV_PHASE_BIAS_DEG"]),
        c3["AZ_SIGN"],
        c3["EL_SIGN"],
    )
    radar_el = el_msgs[np.abs(times[:, None] - el_ts[None, :]).argmin(axis=1)]

    depth_map_3d = create_lidar_3d(
        bag,
        radar_params,
        R_RL,
        t_RL,
        times,
        pose_gt,
        cfg["LIDAR_TOPIC"],
        cfg["RA"]["RA_MAX"],
        cfg["RA"]["RAMAP_ASIZE"],
        list(c3["HEIGHT_EDGES"]),
        window=c3["MAP_WINDOW"],
        voxel_size=c3["VOXEL_SIZE"],
    )

    out = os.path.splitext(args.bag_path)[0] + "_3d.npz"
    np.savez(out, time=times, radar_el=radar_el, depth_map_3d=depth_map_3d)
    print("radar_el", radar_el.shape, "depth_map_3d", depth_map_3d.shape, "->", out)
    print("occupied fraction per slice:", np.round(depth_map_3d.mean(axis=(0, 2, 3)), 3))
