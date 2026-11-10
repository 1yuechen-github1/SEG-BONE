from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from monai.losses import DiceCELoss

from dataset import load_cases, sample_batch_2d, sample_batch_3d
from evaluate import evaluate
from models import MODELS_2D, MODELS_25D, build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.cases_root)
    train_cases = [c for c in cases if c.split == "train"]

    name = config["model"]
    patch = tuple(config["patch"])
    model = build_model(name, patch).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        model.train()
        losses = []
        for _ in range(max(64, len(train_cases) * 8)):
            if name in MODELS_2D or name in MODELS_25D:
                x, y = sample_batch_2d(train_cases, config["batch"], device, 3 if name in MODELS_25D else 1)
            else:
                x, y = sample_batch_3d(train_cases, config["batch"], patch, device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        print(f"{name} epoch {epoch + 1}/{config['epochs']} loss={np.mean(losses):.4f}", flush=True)

    torch.save(model.state_dict(), args.output_dir / f"{name}.pt")
    evaluate(model, config, cases, args.output_dir, device)


if __name__ == "__main__":
    main()
