from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from monai.losses import DiceCELoss
from tqdm import tqdm

from dataset import load_cases, sample_batch_2d, sample_batch_3d
from evaluate import dice_score, evaluate, predict_volume
from models import MODELS_2D, MODELS_25D, build_model


def setup_logging(log_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    log_dir.mkdir(parents=True, exist_ok=True)
    file_h = logging.FileHandler(log_dir / "log.txt", encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_h)
    return logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint_last.pt")
    parser.add_argument("--cache-dir", type=Path, default=None, help="dir for preprocessed .pt cache")
    parser.add_argument("--refresh-cache", action="store_true", help="rebuild the .pt cache from raw NIfTI")
    parser.add_argument("--val-every", type=int, default=1, help="compute val dice every N epochs")
    parser.add_argument("--log-every", type=int, default=100, help="write per-step loss to log every N steps (0=off)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.output_dir)

    config = json.loads(args.config.read_text())
    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    name = config["model"]
    patch = tuple(config["patch"])

    cache_dir = args.cache_dir or (args.cases_root / "_cache")
    if args.refresh_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)

    cases = load_cases(args.cases_root, cache_dir, normalize=config.get("normalize", True))
    train_cases = [c for c in cases if c.split == "train"]
    val_cases = [c for c in cases if c.split == "val"]
    test_cases = [c for c in cases if c.split == "test"]

    log.info(f"config file: {args.config}")
    log.info(f"config: {json.dumps(config, ensure_ascii=False)}")
    log.info(f"cases-root: {args.cases_root} | cache-dir: {cache_dir}")
    log.info(f"dataset: train={len(train_cases)} val={len(val_cases)} test={len(test_cases)}")
    log.info(f"device={device} model={name} patch={patch} batch={config['batch']} lr={config['lr']} epochs={config['epochs']}")

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
        log.info(f"resumed from epoch {start_epoch}, best val dice={best_dice:.4f}")
    elif args.resume:
        log.info("no checkpoint found, starting from scratch")

    steps_per_epoch = max(64, len(train_cases) * 8)
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        losses = []
        ema = 0.0
        pbar = tqdm(range(steps_per_epoch), desc=f"{name} epoch {epoch + 1}/{config['epochs']}", ncols=100)
        for step in pbar:
            if name in MODELS_2D or name in MODELS_25D:
                x, y = sample_batch_2d(train_cases, config["batch"], device, 3 if name in MODELS_25D else 1)
            else:
                x, y = sample_batch_3d(train_cases, config["batch"], patch, device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            loss_val = float(loss)
            losses.append(loss_val)
            ema = ema * 0.95 + loss_val * 0.05 if ema else loss_val
            pbar.set_postfix(loss=f"{loss_val:.4f}", avg=f"{ema:.4f}")
            if args.log_every and (step + 1) % args.log_every == 0:
                log.debug(f"epoch {epoch + 1} step {step + 1}/{steps_per_epoch} loss={loss_val:.4f} avg={ema:.4f}")
        train_loss = float(np.mean(losses))

        val_dice = None
        if val_cases and (epoch + 1) % args.val_every == 0:
            dices = []
            for c in tqdm(val_cases, desc="  val", leave=False, ncols=100):
                dices.append(dice_score(predict_volume(model, c, name, patch, device), c.mask))
            val_dice = float(np.mean(dices))
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), best_path)
                log.info(f"new best model saved -> {best_path.name} (val dice={best_dice:.4f})")

        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_dice": best_dice},
            ckpt_path,
        )

        msg = f"epoch {epoch + 1}/{config['epochs']} train_loss={train_loss:.4f}"
        if val_dice is not None:
            msg += f" val_dice={val_dice:.4f}"
        msg += f" best_dice={best_dice:.4f}"
        log.info(msg)

    torch.save(model.state_dict(), args.output_dir / f"{name}_last.pt")
    log.info(f"training finished, final model -> {name}_last.pt")
    rows = evaluate(model, config, cases, args.output_dir, device)
    for split in ("val", "test"):
        sub = [r for r in rows if r["split"] == split]
        if sub:
            log.info(f"final {split} dice mean={np.mean([float(r['dice']) for r in sub]):.4f} (n={len(sub)})")
    log.info(f"final best val dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
