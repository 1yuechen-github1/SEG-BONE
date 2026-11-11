from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference

from dataset import Case, load_cases
from models import MODELS_2D, MODELS_25D, build_model


def predict_volume(model, case: Case, name: str, patch: tuple[int, int, int], device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        if name in MODELS_2D or name in MODELS_25D:
            channels = 3 if name in MODELS_25D else 1
            slices = []
            for z in range(case.ct.shape[0]):
                if channels == 1:
                    plane = case.ct[z][None]
                else:
                    plane = torch.stack([case.ct[max(0, z - 1)], case.ct[z], case.ct[min(case.ct.shape[0] - 1, z + 1)]])
                slices.append(plane)
            batch = torch.stack(slices)[:, None].float().to(device)
            probs = []
            for i in range(0, len(batch), 16):
                probs.append(torch.softmax(model(batch[i:i + 16]), dim=1).cpu().numpy())
            prob = np.concatenate(probs)[:, 1]
            return prob.transpose(1, 2, 0) > 0.5
        volume = case.ct[None, None].float().to(device)
        prob = sliding_window_inference(volume, patch, 4, model, overlap=0.5)
        return torch.softmax(prob, dim=1)[0, 1].cpu().numpy() > 0.5


def _as_bool_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x).astype(bool)


def dice_score(first, second) -> float:
    first = _as_bool_numpy(first)
    second = _as_bool_numpy(second)
    union = int((first | second).sum())
    return 1.0 if union == 0 else 2.0 * int((first & second).sum()) / union


def icc31(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    row_means = (x + y) / 2
    grand = row_means.mean()
    msr = 2 * ((row_means - grand) ** 2).sum() / (n - 1)
    mse = (((x - row_means) ** 2).sum() + ((y - row_means) ** 2).sum()) / (n - 1)
    return (msr - mse) / (msr + mse)


def save_pred_nifti(pred: np.ndarray, case: Case, out_path: Path) -> None:
    img = sitk.GetImageFromArray(pred.astype(np.uint8))
    img.SetSpacing(tuple(float(s) for s in case.spacing))
    img.SetOrigin(tuple(float(o) for o in case.origin))
    img.SetDirection(tuple(float(d) for d in case.direction))
    sitk.WriteImage(img, str(out_path))


def evaluate(model, config: dict, cases: list[Case], out_dir: Path, device) -> list[dict]:
    name = config["model"]
    patch = tuple(config["patch"])
    rows = []
    for case in cases:
        if case.split == "train":
            continue
        pred = predict_volume(model, case, name, patch, device)
        voxel_ml = float(np.prod(case.spacing)) / 1000.0
        rows.append({
            "case_id": case.case_id,
            "split": case.split,
            "model": name,
            "dice": dice_score(pred, case.mask),
            "pred_vol_ml": float(pred.sum()) * voxel_ml,
            "gt_vol_ml": float(case.mask.sum()) * voxel_ml,
        })
        pred_dir = out_dir / "predictions" / name / case.split
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.save(pred_dir / f"{case.case_id}.npy", pred.astype(np.uint8))
        save_pred_nifti(pred, case, pred_dir / f"{case.case_id}.nii.gz")
    with (out_dir / f"eval_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config["model"], tuple(config["patch"])).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.cases_root)
    evaluate(model, config, cases, args.output_dir, device)


if __name__ == "__main__":
    main()
