from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from tqdm import tqdm


def find_pair(case_dir: Path) -> tuple[Path, Path]:
    images = sorted(case_dir.glob("*_ct*.n*")) or sorted(case_dir.glob("*image*.n*"))
    masks = sorted(case_dir.glob("*_mask*.n*")) or sorted(case_dir.glob("*label*.n*")) or sorted(case_dir.glob("*_gt*.n*"))
    return images[0], masks[0]


def normalize_ct(ct: np.ndarray, lower: float = 0.5, upper: float = 99.5) -> np.ndarray:
    """Intensity clip to [lower, upper] percentiles, then z-score. Returns float32."""
    lo, hi = np.percentile(ct, [lower, upper])
    ct = np.clip(ct, lo, hi)
    return ((ct - ct.mean()) / (ct.std() + 1e-8)).astype(np.float32)


class Case:
    def __init__(self, case_id: str, split: str, ct, mask, spacing, fg):
        self.case_id = case_id
        self.split = split
        self.ct = ct if isinstance(ct, torch.Tensor) else torch.as_tensor(ct)
        self.mask = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
        self.fg = fg if isinstance(fg, torch.Tensor) else torch.as_tensor(fg)
        self.spacing = np.asarray(spacing, dtype=float)

    @classmethod
    def from_dir(cls, case_dir: Path, split: str, normalize: bool = True) -> "Case":
        ct_path, mask_path = find_pair(case_dir)
        ct_img = sitk.ReadImage(str(ct_path))
        spacing = np.asarray(ct_img.GetSpacing(), dtype=float)
        ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) > 0
        if normalize:
            ct = normalize_ct(ct)
        fg = np.argwhere(mask)
        return cls(case_dir.name, split, ct, mask, spacing, fg)


def _cache_path(cache_dir: Path, split: str, case_id: str) -> Path:
    return cache_dir / split / f"{case_id}.pt"


def _save_cache(cache_dir: Path, case: Case) -> None:
    path = _cache_path(cache_dir, case.split, case.case_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"ct": case.ct, "mask": case.mask, "spacing": torch.as_tensor(case.spacing), "fg": case.fg},
            path,
        )
    except OSError as exc:
        print(f"warning: could not write cache {path}: {exc}", flush=True)


def _load_cache(cache_dir: Path, split: str, case_id: str) -> Case | None:
    path = _cache_path(cache_dir, split, case_id)
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu")
    spacing = data["spacing"]
    if torch.is_tensor(spacing):
        spacing = spacing.cpu().numpy()
    return Case(case_id, split, data["ct"], data["mask"], spacing, data["fg"])


def load_cases(cases_root: Path, cache_dir: Path | None = None, normalize: bool = True) -> list[Case]:
    """Load all cases, using (and building) a per-case .pt cache to skip NIfTI decompression."""
    cache_dir = cache_dir or (cases_root / "_cache")
    entries = [
        (split, case_dir)
        for split in ("train", "val", "test")
        if (cases_root / split).is_dir()
        for case_dir in sorted((cases_root / split).iterdir())
        if case_dir.is_dir()
    ]
    cases = []
    for split, case_dir in tqdm(entries, desc="load cases", ncols=100):
        case = _load_cache(cache_dir, split, case_dir.name)
        if case is None:
            case = Case.from_dir(case_dir, split, normalize)
            _save_cache(cache_dir, case)
        cases.append(case)
    return cases


def crop_or_pad(volume: torch.Tensor, center: tuple[int, int, int], patch: tuple[int, int, int]) -> torch.Tensor:
    pads = [
        max(0, patch[i] // 2 - center[i], center[i] + patch[i] // 2 + 1 - volume.shape[i])
        for i in range(3)
    ]
    if any(pads):
        volume = F.pad(volume, (pads[2], pads[2], pads[1], pads[1], pads[0], pads[0]), mode="replicate")
        center = tuple(c + p for c, p in zip(center, pads))
    sl = tuple(slice(center[i] - patch[i] // 2, center[i] - patch[i] // 2 + patch[i]) for i in range(3))
    return volume[sl]


def sample_batch_3d(cases: list[Case], batch: int, patch: tuple[int, int, int], device) -> tuple:
    xs, ys = [], []
    for _ in range(batch):
        case = cases[np.random.randint(len(cases))]
        if len(case.fg) and np.random.rand() < 0.5:
            center = tuple(int(v) for v in case.fg[np.random.randint(len(case.fg))])
        else:
            center = tuple(int(np.random.randint(0, s)) for s in case.ct.shape)
        xs.append(crop_or_pad(case.ct, center, patch)[None])
        ys.append(crop_or_pad(case.mask.float(), center, patch)[None])
    return (
        torch.stack(xs).to(device),
        torch.stack(ys).long().to(device),
    )


def sample_batch_2d(cases: list[Case], batch: int, device, channels: int) -> tuple:
    xs, ys = [], []
    for _ in range(batch):
        case = cases[np.random.randint(len(cases))]
        z_slices = torch.nonzero(case.mask.any(dim=(1, 2))).squeeze(1)
        if len(z_slices) and np.random.rand() < 0.5:
            z = int(z_slices[np.random.randint(len(z_slices))])
        else:
            z = int(np.random.randint(case.ct.shape[0]))
        if channels == 1:
            xs.append(case.ct[z][None])
        else:
            xs.append(torch.stack([case.ct[max(0, z - 1)], case.ct[z], case.ct[min(case.ct.shape[0] - 1, z + 1)]]))
        ys.append(case.mask[z])
    return (
        torch.stack(xs)[:, None].float().to(device),
        torch.stack(ys)[:, None].long().to(device),
    )
