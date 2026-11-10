from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate import icc31


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = []
    for path in sorted(args.output_dir.glob("eval_*.csv")):
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for split in ("val", "test"):
            sub = [r for r in rows if r["split"] == split]
            if sub:
                summary.append({
                    "model": rows[0]["model"],
                    "split": split,
                    "n": len(sub),
                    "dice_mean": float(np.mean([float(r["dice"]) for r in sub])),
                    "vol_icc": icc31(
                        [float(r["pred_vol_ml"]) for r in sub],
                        [float(r["gt_vol_ml"]) for r in sub],
                    ),
                })
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "split", "n", "dice_mean", "vol_icc"])
        writer.writeheader()
        writer.writerows(summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
