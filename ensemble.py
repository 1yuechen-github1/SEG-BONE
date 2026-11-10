from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from dataset import load_cases
from evaluate import dice_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    names = args.models.split(",")
    cases = load_cases(args.cases_root)
    rows = []
    for case in cases:
        if case.split == "train":
            continue
        votes = None
        for name in names:
            pred = np.load(args.output_dir / "predictions" / name / case.split / f"{case.case_id}.npy") > 0
            votes = pred.astype(np.uint8) if votes is None else votes + pred
        pred = votes * 2 > len(names)
        voxel_ml = float(np.prod(case.spacing)) / 1000.0
        rows.append({
            "case_id": case.case_id,
            "split": case.split,
            "model": "+".join(names),
            "dice": dice_score(pred, case.mask),
            "pred_vol_ml": float(pred.sum()) * voxel_ml,
            "gt_vol_ml": float(case.mask.sum()) * voxel_ml,
        })
    out_name = f"eval_{'_'.join(names)}.csv"
    with (args.output_dir / out_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
