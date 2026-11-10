from __future__ import annotations

import torch
import segmentation_models_pytorch as smp


def build_smp2d(name: str) -> torch.nn.Module:
    if name == "unetpp_2d":
        return smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=2)
    if name == "unetpp_25d":
        return smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=2)
    if name == "deeplabv3p_2d":
        return smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=2)
    raise KeyError(name)
