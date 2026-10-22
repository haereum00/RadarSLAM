#!/usr/bin/env python3

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FlipRange3D(object):
    topics = [
        "radar_r_1",
        "radar_r_3",
        "radar_r_5",
        "radar_re_1",
        "radar_re_3",
        "radar_re_5",
        "radar_el",
        "depth_map_3d",
    ]

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if torch.rand(1).item() < self.prob:
            for topic in self.topics:
                if topic in sample:
                    sample[topic] = transforms.functional.hflip(sample[topic])
        return sample


class UNet3DDataset(Dataset):
    topics = [
        "time",
        "pose_gt",
        "radar_r_1",
        "radar_r_3",
        "radar_r_5",
        "radar_re_1",
        "radar_re_3",
        "radar_re_5",
    ]
    topics_3d = ["radar_el", "depth_map_3d"]

    def __init__(self, path, transform=None):
        self.path = path
        print(path)
        with np.load(path) as data:
            self.dataset = {k: data[k] for k in data.files if k in self.topics}
        with np.load(path.replace(".npz", "_3d.npz")) as data:
            assert np.array_equal(data["time"], self.dataset["time"]), "frames of _3d.npz do not match"
            self.dataset.update({k: data[k] for k in self.topics_3d})

        lengths = [v.shape[0] for v in self.dataset.values()]
        assert len(set(lengths)) == 1
        self.num_samples = lengths[0]
        self.transform = transform

    def __len__(self):
        return self.num_samples - 1

    def __getitem__(self, idx):
        sample = {"time": torch.tensor(self.dataset["time"][idx : idx + 1])}
        for k in ["radar_r_1", "radar_r_3", "radar_r_5", "radar_re_1", "radar_re_3", "radar_re_5", "radar_el"]:
            sample[k] = torch.from_numpy(np.asarray(self.dataset[k][idx], dtype=np.float32))
        sample["depth_map_3d"] = torch.from_numpy(self.dataset["depth_map_3d"][idx + 1].astype(np.float32))
        sample["pose_gt"] = torch.from_numpy(self.dataset["pose_gt"][idx + 1])
        if self.transform:
            sample = self.transform(sample)
        return sample


def stack_input(batch):
    return torch.cat(
        [
            batch["radar_r_1"],
            batch["radar_r_3"],
            batch["radar_r_5"],
            batch["radar_re_1"],
            batch["radar_re_3"],
            batch["radar_re_5"],
            batch["radar_el"],
        ],
        dim=1,
    ).to(torch.float32)
