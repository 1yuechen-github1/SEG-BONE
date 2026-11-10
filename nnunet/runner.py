from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dice_score(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(bool)
    second = second.astype(bool)
    union = int((first | second).sum())
    return 1.0 if union == 0 else 2.0 * int((first & second).sum()) / union


def find_pair(case_dir: Path) -> tuple[Path, Path]:
    images = sorted(case_dir.glob("*_ct*.n*")) or sorted(case_dir.glob("*image*.n*"))
    masks = sorted(case_dir.glob("*_mask*.n*")) or sorted(case_dir.glob("*label*.n*")) or sorted(case_dir.glob("*_gt*.n*"))
    return images[0], masks[0]


def nnunet_env(base: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_raw"] = str(base / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(base / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(base / "nnUNet_results")
    return env


def convert_dataset(cases_root: Path, base: Path, dataset_id: int) -> dict[str, list[str]]:
    raw = base / "nnUNet_raw" / f"Dataset{dataset_id}_PeriodontalDefect"
    images_tr = raw / "imagesTr"
    labels_tr = raw / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for split in splits:
        for case_dir in sorted((cases_root / split).iterdir()):
            if not case_dir.is_dir():
                continue
            ct_path, mask_path = find_pair(case_dir)
            case_id = case_dir.name
            sitk.WriteImage(sitk.ReadImage(str(ct_path)), str(images_tr / f"{case_id}_0000.nii.gz"))
            sitk.WriteImage(sitk.ReadImage(str(mask_path)), str(labels_tr / f"{case_id}.nii.gz"))
            splits[split].append(case_id)

    write_json(raw / "dataset.json", {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "defect": 1},
        "numTraining": len(splits["train"]) + len(splits["val"]),
        "file_ending": ".nii.gz",
    })
    preprocessed = base / "nnUNet_preprocessed" / f"Dataset{dataset_id}_PeriodontalDefect"
    preprocessed.mkdir(parents=True, exist_ok=True)
    write_json(preprocessed / "splits_final.json", [{"train": splits["train"], "val": splits["val"]}])
    return splits


def run(cases_root: Path, output_dir: Path, seed: int, epochs: int, dataset_id: int, fold: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir.parent
    env = nnunet_env(base)

    splits = convert_dataset(cases_root, base, dataset_id)

    result = subprocess.run(
        ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "-c", "3d_fullres"],
        env=env, capture_output=True, text=True, timeout=3600,
    )
    (output_dir / "preprocess_stdout.log").write_text(result.stdout, encoding="utf-8")
    (output_dir / "preprocess_stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    print(f"[{now()}] preprocessing done")

    train_env = env.copy()
    train_env["NNUNET_RAW_SEED"] = str(seed)
    result = subprocess.run(
        ["nnUNetv2_train", str(dataset_id), "3d_fullres", str(fold),
         "-tr", "nnUNetTrainer", "-p", "nnUNetPlans", "--num_epochs", str(epochs)],
        env=train_env, capture_output=True, text=True,
    )
    (output_dir / "train_stdout.log").write_text(result.stdout, encoding="utf-8")
    (output_dir / "train_stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    print(f"[{now()}] training done")

    images_tr = base / "nnUNet_raw" / f"Dataset{dataset_id}_PeriodontalDefect" / "imagesTr"
    labels_tr = base / "nnUNet_raw" / f"Dataset{dataset_id}_PeriodontalDefect" / "labelsTr"
    eval_rows = []
    for split_name in ("val", "test"):
        for case_id in splits[split_name]:
            pred_dir = output_dir / "predictions" / split_name / case_id
            pred_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["nnUNetv2_predict", "-i", str(images_tr / f"{case_id}_0000.nii.gz"),
                 "-o", str(pred_dir), "-d", str(dataset_id), "-c", "3d_fullres", "-f", str(fold)],
                env=env, capture_output=True, text=True, timeout=1800,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr[-2000:])
            pred_path = next(pred_dir.glob("*.nii.gz"))
            pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))) > 0
            gt = sitk.GetArrayFromImage(sitk.ReadImage(str(labels_tr / f"{case_id}.nii.gz"))) > 0
            eval_rows.append({
                "case_id": case_id,
                "split": split_name,
                "model": "nnunet_v2_repaired",
                "seed": seed,
                "dice": dice_score(pred, gt),
            })

    fields = list(eval_rows[0].keys())
    write_csv(output_dir / "eval_val.csv", [r for r in eval_rows if r["split"] == "val"], fields)
    write_csv(output_dir / "eval_test.csv", [r for r in eval_rows if r["split"] == "test"], fields)

    test_dice = float(np.mean([float(r["dice"]) for r in eval_rows if r["split"] == "test" and r["dice"] == r["dice"]]))
    (output_dir / "DONE").write_text(now() + "\n", encoding="utf-8")
    print(f"[{now()}] nnunet seed {seed} done. test dice={test_dice:.4f}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--dataset-id", type=int, default=701)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    raise SystemExit(run(args.cases_root, args.output_dir, args.seed, args.epochs, args.dataset_id, args.fold))


if __name__ == "__main__":
    main()
