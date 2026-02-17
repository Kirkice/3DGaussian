from __future__ import annotations

import platform

import torch


def get_default_device() -> torch.device:
    if platform.system() == "Windows" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
