from __future__ import annotations

import torch
from monai.networks.nets import AttentionUNet, DynUNet, MedNeXt, SegResNet, SwinUNETR, UNETR


def build_monai3d(name: str, patch: tuple[int, int, int]) -> torch.nn.Module:
    if name == "segresnet_d1":
        return SegResNet(spatial_dims=3, in_channels=1, out_channels=2, init_filters=16)
    if name == "attention_unet_3d":
        return AttentionUNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256))
    if name == "dynunet":
        return DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            kernel_size=[[3, 3, 3]] * 4,
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[[2, 2, 2]] * 3,
            norm_name="instance",
        )
    if name == "unetr":
        return UNETR(in_channels=1, out_channels=2, img_size=patch, feature_size=16)
    if name == "swunetr":
        return SwinUNETR(img_size=patch, in_channels=1, out_channels=2, feature_size=48)
    if name == "mednext_small":
        return MedNeXt(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            init_filters=32,
            num_blocks=[2, 2, 2, 2, 2],
        )
    raise KeyError(name)
