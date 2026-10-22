#!/usr/bin/env python3

import os

import argparse
from collections import defaultdict, deque

import numpy as np
import rosbag
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R, Slerp
from tqdm import tqdm

np.set_printoptions(precision=3, floatmode="fixed", sign=" ")


from radarize.config import cfg, update_config
from radarize.utils import dsp, radar_config

RADAR_CHANNEL_ORDER = None


def read_radar_cube(msg):
    radar_cube = dsp.reshape_frame(msg)
    if RADAR_CHANNEL_ORDER is not None:
        radar_cube = radar_cube[:, RADAR_CHANNEL_ORDER, :]
    return radar_cube


def create_radar_bev(
    bag,
    radar_params,
    radar_topic="/radar0/radar_data",
    radar_buffer_len=1,
    angle_res=1,
    angle_range=43,
    normalization_range=[0.0, 18.0],
):
    heatmap_ts, heatmap_msgs = [], []

    radar_buffer = deque(maxlen=radar_buffer_len)

    for i, (topic, msg, ts) in tqdm(
        enumerate(bag.read_messages([radar_topic])),
        total=bag.get_message_count(radar_topic),
    ):
        radar_cube = read_radar_cube(msg)

        radar_buffer.append(radar_cube)
        if len(radar_buffer) < radar_buffer.maxlen:
            continue
        radar_cube = np.concatenate(radar_buffer, axis=0)

        radar_cube_a = radar_cube[:, :8, :]
        radar_cube_a = dsp._tdm(radar_cube_a, 2, 4)

        heatmap = np.stack(
            [
                dsp.compute_range_azimuth(
                    radar_cube_a, angle_res, angle_range, "capon"
                ),
            ]
        )

        heatmap = dsp.normalize(
            heatmap, min_val=normalization_range[0], max_val=normalization_range[1]
        )

        heatmap_msgs.append(heatmap)
        heatmap_ts.append(ts.secs + 1e-9 * ts.nsecs)

    heatmap_ts = np.array(heatmap_ts)
    return heatmap_ts, heatmap_msgs


def create_radar_bev_elevation(
    bag,
    radar_params,
    radar_topic="/radar0/radar_data",
    radar_buffer_len=1,
    angle_res=1,
    angle_range=43,
    normalization_range=[0.0, 18.0],
):
    heatmap_ts, heatmap_msgs = [], []

    radar_buffer = deque(maxlen=radar_buffer_len)

    for i, (topic, msg, ts) in tqdm(
        enumerate(bag.read_messages([radar_topic])),
        total=bag.get_message_count(radar_topic),
    ):
        radar_cube = read_radar_cube(msg)

        radar_buffer.append(radar_cube)
        if len(radar_buffer) < radar_buffer.maxlen:
            continue
        radar_cube = np.concatenate(radar_buffer, axis=0)

        radar_cube_e = (
            radar_cube[:, 2:6, :] + radar_cube[:, 8:12, :]
        ) / 2
        radar_cube_e = dsp._tdm(radar_cube_e, 2, 4)

        heatmap = np.stack(
            [dsp.compute_range_azimuth(radar_cube_e, angle_res, angle_range, "capon")]
        )

        heatmap = dsp.normalize(
            heatmap, min_val=normalization_range[0], max_val=normalization_range[1]
        )

        heatmap_msgs.append(heatmap)
        heatmap_ts.append(ts.secs + 1e-9 * ts.nsecs)

    heatmap_ts = np.array(heatmap_ts)
    return heatmap_ts, heatmap_msgs


def create_radar_doppler(
    bag,
    radar_params,
    radar_topic="/radar0/radar_data",
    resize_shape=(181, 60),
    radar_buffer_len=3,
    range_subsampling_factor=1,
    angle_res=1,
    angle_range=90,
    normalization_range=[10.0, 25.0],
):
    heatmap_ts, heatmap_msgs = [], []

    radar_buffer = deque(maxlen=radar_buffer_len)

    for i, (topic, msg, ts) in tqdm(
        enumerate(bag.read_messages([radar_topic])),
        total=bag.get_message_count(radar_topic),
    ):
        radar_cube = read_radar_cube(msg)

        radar_buffer.append(radar_cube)
        if len(radar_buffer) < radar_buffer.maxlen:
            continue
        radar_cube = np.concatenate(radar_buffer, axis=0)

        radar_cube_h = radar_cube[::1]
        heatmap_h = dsp.preprocess_1d_radar_1843(
            radar_cube_h,
            angle_res,
            angle_range,
            range_subsampling_factor,
            normalization_range[0],
            normalization_range[1],
            resize_shape,
        )
        heatmap_h = np.fliplr(heatmap_h)

        heatmap = np.stack(
            [
                heatmap_h,
            ]
        )

        heatmap_msgs.append(heatmap)
        heatmap_ts.append(ts.secs + 1e-9 * ts.nsecs)

    heatmap_ts = np.array(heatmap_ts)
    return heatmap_ts, heatmap_msgs


def create_radar_doppler_elevation(
    bag,
    radar_params,
    radar_topic="/radar0/radar_data",
    resize_shape=(181, 60),
    radar_buffer_len=3,
    range_subsampling_factor=1,
    angle_res=1,
    angle_range=90,
    normalization_range=[10.0, 25.0],
):
    heatmap_ts, heatmap_msgs = [], []

    radar_buffer = deque(maxlen=radar_buffer_len)

    for i, (topic, msg, ts) in tqdm(
        enumerate(bag.read_messages([radar_topic])),
        total=bag.get_message_count(radar_topic),
    ):
        radar_cube = read_radar_cube(msg)

        radar_buffer.append(radar_cube)
        if len(radar_buffer) < radar_buffer.maxlen:
            continue
        radar_cube = np.concatenate(radar_buffer, axis=0)

        radar_cube_e = (
            radar_cube[:, 2:6, :] + radar_cube[:, 8:12, :]
        ) / 2

        radar_cube_h = radar_cube_e[::1]
        heatmap_h = dsp.preprocess_1d_radar_1843(
            radar_cube_h,
            angle_res,
            angle_range,
            range_subsampling_factor,
            normalization_range[0],
            normalization_range[1],
            resize_shape,
        )
        heatmap_h = np.fliplr(heatmap_h)

        heatmap = np.stack(
            [
                heatmap_h,
            ]
        )

        heatmap_msgs.append(heatmap)
        heatmap_ts.append(ts.secs + 1e-9 * ts.nsecs)

    heatmap_ts = np.array(heatmap_ts)
    return heatmap_ts, heatmap_msgs


def load_lidar_radar_extrinsic(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"LiDAR-Radar extrinsic not found: {path}")

    if os.path.splitext(path)[1] in [".yaml", ".yml"]:
        with open(path, "r") as f:
            calib = yaml.safe_load(f)
    else:
        calib = dict(np.load(path))

    R_RL = np.asarray(calib["R_RL"], dtype=np.float64).reshape(3, 3)
    t_RL = np.asarray(calib["t_RL"], dtype=np.float64).reshape(3)
    t_IL = np.asarray(calib.get("t_IL", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)

    return R_RL, t_RL, t_IL


def make_transform(rot, trans):
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


def livox_msg_to_numpy(msg):
    t0 = msg.header.stamp.to_sec()

    points = np.array(
        [(p.x, p.y, p.z) for p in msg.points], dtype=np.float64
    ).reshape(-1, 3)
    times = t0 + 1e-9 * np.array(
        [p.offset_time for p in msg.points], dtype=np.float64
    )

    return points, times


def polar_occupancy_map(points, range_axis, angle_axis):
    x_w = len(range_axis)
    y_w = len(angle_axis)
    occupancy_map = np.ones((x_w, y_w), dtype=np.uint8)

    range_p = np.linalg.norm(points, axis=1)
    angle_p = -1 * np.arctan2(points[:, 0], points[:, 1])

    valid = (
        (range_p < range_axis[-1])
        & (angle_p > angle_axis[0])
        & (angle_p < angle_axis[-1])
    )
    range_p = range_p[valid]
    angle_p = angle_p[valid]

    ix = np.rint(
        (range_p - range_axis[0]) / (range_axis[1] - range_axis[0])
    ).astype(int)
    iy = np.rint(
        (angle_p - angle_axis[0]) / (angle_axis[1] - angle_axis[0])
    ).astype(int)

    keep = (ix >= 1) & (ix < x_w - 1) & (iy >= 1) & (iy < y_w - 1)
    ix = ix[keep]
    iy = iy[keep]

    occupancy_map[ix, iy] = 0
    occupancy_map[ix + 1, iy] = 0
    occupancy_map[ix, iy + 1] = 0
    occupancy_map[ix + 1, iy + 1] = 0

    return occupancy_map


def interpolate_poses(pose_ts, T_WR, query_ts):
    query_ts = np.clip(query_ts, pose_ts[0], pose_ts[-1])
    rotations = Slerp(pose_ts, R.from_matrix(T_WR[:, :3, :3]))(query_ts)
    positions = np.stack(
        [np.interp(query_ts, pose_ts, T_WR[:, k, 3]) for k in range(3)], axis=1
    )
    return rotations, positions


def poses_to_msgs(T_WR):
    return np.concatenate(
        [T_WR[:, :3, 3], R.from_matrix(T_WR[:, :3, :3]).as_quat()], axis=1
    )


def create_fastlio_pose_gt(
    bag,
    R_RL,
    t_RL,
    t_IL,
    pose_topic="/Odometry",
    imu_topic="/livox/imu",
):
    T_RL = make_transform(R_RL, t_RL)
    T_IL = make_transform(np.eye(3), t_IL)

    T_IR = T_IL @ np.linalg.inv(T_RL)

    pose_ts = []
    T_WR = []

    for topic, msg, ts in tqdm(
        bag.read_messages([pose_topic]),
        total=bag.get_message_count(pose_topic),
    ):
        pose = msg.pose.pose
        T_WI = make_transform(
            R.from_quat(
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ]
            ).as_matrix(),
            [pose.position.x, pose.position.y, pose.position.z],
        )

        pose_ts.append(msg.header.stamp.to_sec())
        T_WR.append(T_WI @ T_IR)

    pose_ts = np.array(pose_ts)
    T_WR = np.array(T_WR)

    R_WI = Slerp(pose_ts, R.from_matrix(T_WR[:, :3, :3] @ T_IR[:3, :3].T))
    acc, acc_ts = [], []
    for topic, msg, ts in bag.read_messages([imu_topic]):
        t = msg.header.stamp.to_sec()
        if pose_ts[0] <= t <= pose_ts[-1]:
            a = msg.linear_acceleration
            acc.append([a.x, a.y, a.z])
            acc_ts.append(t)

    if acc:
        up_W = np.mean(R_WI(np.array(acc_ts)).apply(np.array(acc)), axis=0)
        up_W /= np.linalg.norm(up_W)
        R_level, _ = R.align_vectors([[0.0, 0.0, 1.0]], [up_W])
        T_WR = make_transform(R_level.as_matrix(), np.zeros(3)) @ T_WR
        print(
            f"Leveled world frame by {np.degrees(np.arccos(np.clip(up_W[2], -1, 1))):.2f} deg"
        )
    else:
        print(f"WARNING: no {imu_topic} messages, world frame not leveled.")

    yaw0 = np.arctan2(T_WR[0, 1, 0], T_WR[0, 0, 0])
    T_WR[:, :3, 3] -= T_WR[0, :3, 3]
    T_WR = make_transform(R.from_euler("z", -yaw0).as_matrix(), np.zeros(3)) @ T_WR

    return pose_ts, T_WR


def create_fastlio_body_velo_gt(pose_ts, T_WR):
    body_velo_gt_ts, body_velo_gt_msgs = [], []

    for i in range(1, len(pose_ts)):
        elapsed = pose_ts[i] - pose_ts[i - 1]
        if elapsed <= 0:
            continue

        velo = (T_WR[i, :3, 3] - T_WR[i - 1, :3, 3]) / elapsed
        body_velo = T_WR[i - 1, :3, :3].T @ velo

        if body_velo_gt_msgs and np.linalg.norm(body_velo - body_velo_gt_msgs[-1]) > 0.5:
            body_velo = body_velo_gt_msgs[-1]

        body_velo_gt_msgs.append(body_velo)
        body_velo_gt_ts.append(pose_ts[i])

    return np.array(body_velo_gt_ts), np.array(body_velo_gt_msgs)


def create_lidar_bev(
    bag,
    radar_params,
    R_RL,
    t_RL,
    pose_ts,
    T_WR,
    lidar_topic="/livox/lidar",
    angle_range=43,
    angle_bins_list=(88,),
    z_half=0.05,
    min_range=0.2,
    map_band=0.5,
    voxel_size=0.03,
    window=15.0,
    time_bucket=2.0,
    max_lidar_range=15.0,
):
    range_max = radar_params["range_max"]
    range_bins = radar_params["n_samples"]

    T_RL = make_transform(R_RL, t_RL)

    map_points, map_ts = [], []

    for topic, msg, ts in tqdm(
        bag.read_messages([lidar_topic]),
        total=bag.get_message_count(lidar_topic),
    ):
        pcd, pcd_ts = livox_msg_to_numpy(msg)

        dist = np.linalg.norm(pcd, axis=1)
        keep = (
            np.isfinite(pcd).all(axis=1)
            & (dist > 0.3)
            & (dist < max_lidar_range)
            & (pcd_ts >= pose_ts[0])
            & (pcd_ts <= pose_ts[-1])
        )
        pcd, pcd_ts = pcd[keep], pcd_ts[keep]
        if len(pcd) == 0:
            continue

        pcd = pcd @ T_RL[:3, :3].T + T_RL[:3, 3]
        rotations, positions = interpolate_poses(pose_ts, T_WR, pcd_ts)
        pcd = rotations.apply(pcd) + positions

        keep = np.abs(pcd[:, 2] - positions[:, 2]) < map_band
        pcd, pcd_ts = pcd[keep], pcd_ts[keep]
        if len(pcd) == 0:
            continue

        keys = np.hstack(
            [
                np.floor(pcd / voxel_size).astype(np.int64),
                np.floor(pcd_ts / time_bucket).astype(np.int64)[:, None],
            ]
        )
        _, idxs = np.unique(keys, axis=0, return_index=True)
        map_points.append(pcd[idxs])
        map_ts.append(pcd_ts[idxs])

    map_points = np.concatenate(map_points)
    map_ts = np.concatenate(map_ts)

    keys = np.hstack(
        [
            np.floor(map_points / voxel_size).astype(np.int64),
            np.floor(map_ts / time_bucket).astype(np.int64)[:, None],
        ]
    )
    _, idxs = np.unique(keys, axis=0, return_index=True)
    map_points, map_ts = map_points[idxs], map_ts[idxs]
    print(f"LiDAR map: {len(map_points)} points")

    tree = cKDTree(map_points[:, :2])

    range_axis = np.linspace(0, range_max, range_bins)
    angle_axes = [
        np.linspace(np.deg2rad(-angle_range), np.deg2rad(angle_range), angle_bins)
        for angle_bins in angle_bins_list
    ]

    lidar_maps = [[] for _ in angle_bins_list]

    for i in tqdm(range(len(pose_ts))):
        R_WR = T_WR[i, :3, :3]
        p_WR = T_WR[i, :3, 3]

        idxs = np.array(tree.query_ball_point(p_WR[:2], range_max), dtype=np.int64)
        idxs = idxs[np.abs(map_ts[idxs] - pose_ts[i]) <= window]

        pcd = (map_points[idxs] - p_WR) @ R_WR

        keep = (np.abs(pcd[:, 2]) < z_half) & (
            np.linalg.norm(pcd[:, :2], axis=1) > min_range
        )
        pcd = pcd[keep]

        pcd = np.stack([-pcd[:, 1], pcd[:, 0]], axis=1)

        for k, angle_axis in enumerate(angle_axes):
            lidar_maps[k].append(
                polar_occupancy_map(pcd, range_axis, angle_axis)[None]
            )

    return pose_ts, [np.array(m) for m in lidar_maps]


def sync2topic(unpacked_bag, sync_topic):
    d = defaultdict(lambda: [])
    time_begin = unpacked_bag["pose_gt"][0][0]

    for sync_ts in unpacked_bag[sync_topic][0]:
        if sync_ts < time_begin:
            continue
        for topic, (topic_ts, topic_msgs) in unpacked_bag.items():
            i_topic = np.argmin(np.abs(sync_ts - topic_ts))
            d[topic].append(topic_msgs[i_topic])
        d["time"].append(sync_ts)

    for k, v in d.items():
        d[k] = np.stack(v)

    return d


def args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg", help="experiment configure file name", default=None, type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--bag_path", help="Path to bag directory.", required=True)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = args()
    update_config(cfg, args)
    cfg = cfg["DATASET"]

    print(f"Processing {args.bag_path}...")

    with open(cfg["RADAR_CONFIG"], "r") as f:
        radar_config = radar_config.RadarConfig(f.readlines())
    radar_params = radar_config.get_params()

    bag = rosbag.Bag(args.bag_path)
    R_RL, t_RL, t_IL = load_lidar_radar_extrinsic(cfg["LIDAR_RADAR_EXTRINSICS"])

    if cfg["RADAR_CHANNEL_ORDER"]:
        RADAR_CHANNEL_ORDER = list(cfg["RADAR_CHANNEL_ORDER"])

    da_ts, da_heatmaps = create_radar_doppler(
        bag,
        radar_params,
        radar_topic=cfg["RADAR_TOPIC"],
        resize_shape=cfg["DA"]["RESIZE_SHAPE"],
        radar_buffer_len=cfg["DA"]["RADAR_BUFFER_LEN"],
        range_subsampling_factor=cfg["DA"]["RANGE_SUBSAMPLING_FACTOR"],
    )

    da_e_ts, da_e_heatmaps = create_radar_doppler_elevation(
        bag,
        radar_params,
        radar_topic=cfg["RADAR_TOPIC"],
        resize_shape=cfg["DA"]["RESIZE_SHAPE"],
        radar_buffer_len=cfg["DA"]["RADAR_BUFFER_LEN"],
        range_subsampling_factor=cfg["DA"]["RANGE_SUBSAMPLING_FACTOR"],
    )

    ra_1_ts, ra_1_heatmaps = create_radar_bev(
        bag,
        radar_params,
        radar_topic=cfg["RADAR_TOPIC"],
        radar_buffer_len=1,
        angle_range=cfg["RA"]["RA_MAX"],
    )
    ra_3_ts, ra_3_heatmaps = create_radar_bev(
        bag,
        radar_params,
        radar_topic=cfg["RADAR_TOPIC"],
        radar_buffer_len=3,
        angle_range=cfg["RA"]["RA_MAX"],
    )
    ra_5_ts, ra_5_heatmaps = create_radar_bev(
        bag,
        radar_params,
        radar_topic=cfg["RADAR_TOPIC"],
        radar_buffer_len=5,
        angle_range=cfg["RA"]["RA_MAX"],
    )

    ra_e_1_ts, ra_e_1_heatmaps = create_radar_bev_elevation(
        bag,
        radar_params,
        radar_topic=cfg["RADAR_TOPIC"],
        radar_buffer_len=1,
        angle_range=cfg["RA"]["RA_MAX"],
    )

    ra_e_3_ts, ra_e_3_heatmaps = create_radar_bev_elevation(
        bag,
        radar_params,
        radar_topic=cfg["RADAR_TOPIC"],
        radar_buffer_len=3,
        angle_range=cfg["RA"]["RA_MAX"],
    )

    ra_e_5_ts, ra_e_5_heatmaps = create_radar_bev_elevation(
        bag,
        radar_params,
        radar_topic=cfg["RADAR_TOPIC"],
        radar_buffer_len=5,
        angle_range=cfg["RA"]["RA_MAX"],
    )

    pose_ts, T_WR = create_fastlio_pose_gt(
        bag,
        R_RL,
        t_RL,
        t_IL,
        pose_topic=cfg["LIDAR_ODOM_TOPIC"],
        imu_topic=cfg["LIDAR_IMU_TOPIC"],
    )
    pose_msgs = poses_to_msgs(T_WR)

    body_velo_ts, body_velo_msgs = create_fastlio_body_velo_gt(pose_ts, T_WR)

    depth_ts, (depth_maps, depth_maps_u) = create_lidar_bev(
        bag,
        radar_params,
        R_RL,
        t_RL,
        pose_ts,
        T_WR,
        lidar_topic=cfg["LIDAR_TOPIC"],
        angle_range=cfg["RA"]["RA_MAX"],
        angle_bins_list=(cfg["RA"]["RAMAP_ASIZE"], cfg["RA"]["RAMAP_ASIZE"] * 8),
    )
    depth_u_ts = depth_ts

    frames = defaultdict(lambda: [])

    frames["pose_gt"] = [pose_ts, pose_msgs]
    frames["velo_gt"] = [body_velo_ts, body_velo_msgs]
    frames["depth_map"] = [depth_ts, depth_maps]
    frames["depth_map_u"] = [depth_u_ts, depth_maps_u]
    frames["radar_d"] = [da_ts, da_heatmaps]
    frames["radar_de"] = [da_e_ts, da_e_heatmaps]
    frames["radar_r_1"] = [ra_1_ts, ra_1_heatmaps]
    frames["radar_r_3"] = [ra_3_ts, ra_3_heatmaps]
    frames["radar_r_5"] = [ra_5_ts, ra_5_heatmaps]
    frames["radar_re_1"] = [ra_e_1_ts, ra_e_1_heatmaps]
    frames["radar_re_3"] = [ra_e_3_ts, ra_e_3_heatmaps]
    frames["radar_re_5"] = [ra_e_5_ts, ra_e_5_heatmaps]

    for k, v in frames.items():
        print(k, v[0].shape)

    frames = {k: v for k, v in frames.items() if len(v[0]) > 0}

    synced_bag = sync2topic(frames, cfg["SYNC_TOPIC"])

    for k, v in synced_bag.items():
        print(k, v.shape, v.dtype)

    for k, v in synced_bag.items():
        if v.dtype == np.float64 and k not in ("time", "pose_gt", "velo_gt"):
            synced_bag[k] = v.astype(np.float32)

    np.savez(os.path.splitext(args.bag_path)[0] + ".npz", **synced_bag)
