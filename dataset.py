from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch


def find_pair(case_dir: Path) -> tuple[Path, Path]:
    images = sorted(case_dir.glob("*_ct*.n*")) or sorted(case_dir.glob("*image*.n*"))
    masks = sorted(case_dir.glob("*_mask*.n*")) or sorted(case_dir.glob("*label*.n*")) or sorted(case_dir.glob("*_gt*.n*"))
    return images[0], masks[0]


class Case:
    def __init__(self, case_dir: Path, split: str):
        ct_path, mask_path = find_pair(case_dir)
        ct = sitk.ReadImage(str(ct_path))
        self.case_id = case_dir.name
        self.split = split
        self.spacing = np.array(ct.GetSpacing())
        self.ct = sitk.GetArrayFromImage(ct).astype(np.float32)
        self.mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) > 0
        self.fg = np.argwhere(self.mask)


def load_cases(cases_root: Path) -> list[Case]:
    return [
        Case(case_dir, split)
        for split in ("train", "val", "test")
        for case_dir in sorted((cases_root / split).iterdir())
        if case_dir.is_dir()
    ]


def crop_or_pad(volume: np.ndarray, center: tuple[int, int, int], patch: tuple[int, int, int]) -> np.ndarray:
    pads = [
        max(0, patch[i] // 2 - center[i], center[i] + patch[i] // 2 + 1 - volume.shape[i])
        for i in range(3)
    ]
    if any(pads):
        volume = np.pad(volume, [(p, p) for p in pads], mode="edge")
        center = tuple(c + p for c, p in zip(center, pads))
    sl = tuple(slice(center[i] - patch[i] // 2, center[i] - patch[i] // 2 + patch[i]) for i in range(3))
    return volume[sl]


def sample_batch_3d(cases: list[Case], batch: int, patch: tuple[int, int, int], device) -> tuple:
    xs, ys = [], []
    for _ in range(batch):
        case = cases[np.random.randint(len(cases))]
        if len(case.fg) and np.random.rand() < 0.5:
            center = tuple(case.fg[np.random.randint(len(case.fg))])
        else:
            center = tuple(np.random.randint(0, s) for s in case.ct.shape)
        xs.append(crop_or_pad(case.ct, center, patch)[None])
        ys.append(crop_or_pad(case.mask.astype(np.int8), center, patch)[None])
    return (
        torch.from_numpy(np.stack(xs)).to(device),
        torch.from_numpy(np.stack(ys)).long().to(device),
    )


def sample_batch_2d(cases: list[Case], batch: int, device, channels: int) -> tuple:
    xs, ys = [], []
    for _ in range(batch):
        case = cases[np.random.randint(len(cases))]
        z_slices = np.flatnonzero(case.mask.any(axis=(1, 2)))
        if len(z_slices) and np.random.rand() < 0.5:
            z = int(np.random.choice(z_slices))
        else:
            z = int(np.random.randint(case.ct.shape[0]))
        if channels == 1:
            xs.append(case.ct[z][None])
        else:
            xs.append(np.stack([case.ct[max(0, z - 1)], case.ct[z], case.ct[min(case.ct.shape[0] - 1, z + 1)]]))
        ys.append(case.mask[z].astype(np.int8))
    return (
        torch.from_numpy(np.stack(xs)[:, None]).float().to(device),
        torch.from_numpy(np.stack(ys)[:, None]).long().to(device),
    )
