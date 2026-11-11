from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from monai.losses import DiceCELoss

from dataset import load_cases, sample_batch_2d, sample_batch_3d
from evaluate import dice_score, evaluate, predict_volume
from models import MODELS_2D, MODELS_25D, build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint_last.pt")
    parser.add_argument("--cache-dir", type=Path, default=None, help="dir for preprocessed .pt cache (default cases-root/_cache)")
    parser.add_argument("--refresh-cache", action="store_true", help="rebuild the .pt cache from raw NIfTI")
    parser.add_argument("--val-every", type=int, default=1, help="compute val dice every N epochs")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir or (args.cases_root / "_cache")
    if args.refresh_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)

    cases = load_cases(args.cases_root, cache_dir, normalize=config.get("normalize", True))
    train_cases = [c for c in cases if c.split == "train"]
    val_cases = [c for c in cases if c.split == "val"]
    print(f"loaded {len(cases)} cases (train={len(train_cases)}, val={len(val_cases)})", flush=True)

    name = config["model"]
    patch = tuple(config["patch"])
    model = build_model(name, patch).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    ckpt_path = args.output_dir / "checkpoint_last.pt"
    best_path = args.output_dir / f"{name}_best.pt"
    best_dice = -1.0
    start_epoch = 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        best_dice = float(ck.get("best_dice", -1.0))
        print(f"resumed from epoch {start_epoch}, best val dice={best_dice:.4f}", flush=True)
    elif args.resume:
        print("no checkpoint found, starting from scratch", flush=True)

    steps_per_epoch = max(64, len(train_cases) * 8)
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            if name in MODELS_2D or name in MODELS_25D:
                x, y = sample_batch_2d(train_cases, config["batch"], device, 3 if name in MODELS_25D else 1)
            else:
                x, y = sample_batch_3d(train_cases, config["batch"], patch, device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        train_loss = float(np.mean(losses))

        val_dice = None
        if val_cases and (epoch + 1) % args.val_every == 0:
            val_dice = float(np.mean([dice_score(predict_volume(model, c, name, patch, device), c.mask) for c in val_cases]))
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), best_path)

        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_dice": best_dice},
            ckpt_path,
        )

        msg = f"{name} epoch {epoch + 1}/{config['epochs']} loss={train_loss:.4f}"
        if val_dice is not None:
            msg += f" val_dice={val_dice:.4f}"
        msg += f" best_dice={best_dice:.4f}"
        print(msg, flush=True)

    torch.save(model.state_dict(), args.output_dir / f"{name}_last.pt")
    evaluate(model, config, cases, args.output_dir, device)


if __name__ == "__main__":
    main()
