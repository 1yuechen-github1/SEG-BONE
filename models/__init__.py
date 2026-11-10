from __future__ import annotations

from .monai3d import build_monai3d
from .smp2d import build_smp2d

MODELS_2D = {"unetpp_2d", "deeplabv3p_2d"}
MODELS_25D = {"unetpp_25d"}


def build_model(name: str, patch: tuple[int, int, int]):
    if name in MODELS_2D or name in MODELS_25D:
        return build_smp2d(name)
    return build_monai3d(name, patch)
