"""GolfDB dataset utilities for training and evaluation.

This module exposes:
- GolfDB: torch Dataset for loading preprocessed GolfDB clips.
- ToTensor: transform to convert ndarray frames to float tensors.
- Normalize: transform using channel-wise mean/std.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class GolfDB(Dataset):
    """Golf swing event dataset backed by preprocessed MP4 clips and pickle annotations."""

    def __init__(
        self,
        data_file: str | Path,
        vid_dir: str | Path,
        seq_length: int,
        transform: Optional[object] = None,
        train: bool = True,
        subset_size: Optional[int] = None,
    ) -> None:
        """
        Args:
            data_file: Pickle file containing annotations (id, events).
            vid_dir: Directory containing preprocessed MP4 clips.
            seq_length: Number of frames to sample per clip in train mode.
            transform: Optional transform applied to dict {'images', 'labels'}.
            train: If True, sample random subsequences; else return full clip.
            subset_size: If set, limit dataset to first N entries (debug mode).
        """
        self.df = pd.read_pickle(data_file)
        if subset_size is not None:
            self.df = self.df.iloc[:subset_size].reset_index(drop=True)
        self.vid_dir = Path(vid_dir)
        self.seq_length = seq_length
        self.transform = transform
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        ann = self.df.loc[idx, :]  # annotation info
        events = ann["events"]
        events -= events[0]  # frame numbers correspond to preprocessed clip

        images, labels = [], []
        cap = cv2.VideoCapture(str(self.vid_dir / f"{ann['id']}.mp4"))

        if self.train:
            # random starting position, sample 'seq_length' frames
            start_frame = np.random.randint(events[-1] + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            pos = start_frame
            while len(images) < self.seq_length:
                ret, img = cap.read()
                if ret:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    if pos in events[1:-1]:
                        labels.append(np.where(events[1:-1] == pos)[0][0])
                    else:
                        labels.append(8)
                    pos += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pos = 0
            cap.release()
        else:
            # full clip
            for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                _, img = cap.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                if pos in events[1:-1]:
                    labels.append(np.where(events[1:-1] == pos)[0][0])
                else:
                    labels.append(8)
            cap.release()

        sample = {"images": np.asarray(images), "labels": np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor:
    """Convert ndarrays in sample to PyTorch tensors (N, H, W, C) -> (N, C, H, W)."""

    def __call__(self, sample: dict) -> dict:
        images, labels = sample["images"], sample["labels"]
        images = images.transpose((0, 3, 1, 2))
        return {
            "images": torch.from_numpy(images).float().div(255.0),
            "labels": torch.from_numpy(labels).long(),
        }


class Normalize:
    """Channel-wise normalization using mean/std tensors."""

    def __init__(self, mean, std) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample: dict) -> dict:
        images, labels = sample["images"], sample["labels"]
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {"images": images, "labels": labels}
       

