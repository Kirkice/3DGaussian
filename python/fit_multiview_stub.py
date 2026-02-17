from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from device_utils import get_default_device
from torch_renderer import Camera, look_at, perspective, render_gaussians_torch


def _load_targets(targets_dir: Path, width: int, height: int, device: torch.device) -> list[torch.Tensor]:
    paths = sorted(targets_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG files found in {targets_dir}")

    targets: list[torch.Tensor] = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((width, height), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        targets.append(torch.from_numpy(arr).to(device))
    return targets


def _make_orbit_cameras(num_views: int, width: int, height: int, device: torch.device) -> list[Camera]:
    proj = perspective(60.0, width / height, 0.01, 100.0, device=device)
    target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)

    cams: list[Camera] = []
    radius = 2.5
    pitch = 0.2
    for i in range(num_views):
        yaw = (2.0 * math.pi * i) / max(1, num_views)
        eye = torch.tensor(
            [
                radius * math.cos(pitch) * math.sin(yaw),
                radius * math.sin(pitch),
                radius * math.cos(pitch) * math.cos(yaw),
            ],
            dtype=torch.float32,
            device=device,
        )
        cams.append(Camera(view=look_at(eye, target, up), proj=proj))
    return cams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets_dir", required=True, help="Directory containing target PNG views")
    ap.add_argument("--out_dir", default="outputs/fit_multiview_stub")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--num_gaussians", type=int, default=512)
    args = ap.parse_args()

    device = get_default_device()
    print(f"Using device: {device}")

    targets_dir = Path(args.targets_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = _load_targets(targets_dir, args.width, args.height, device)
    cams = _make_orbit_cameras(len(targets), args.width, args.height, device)

    n = args.num_gaussians
    means = torch.nn.Parameter((torch.rand((n, 3), device=device) - 0.5) * 1.2)
    scales_raw = torch.nn.Parameter(torch.full((n, 3), -2.0, device=device))
    colors_raw = torch.nn.Parameter(torch.rand((n, 3), device=device) * 0.1)
    opacities_raw = torch.nn.Parameter(torch.full((n,), -2.0, device=device))

    opt = torch.optim.Adam([means, scales_raw, colors_raw, opacities_raw], lr=args.lr)
    loss_log: list[float] = []

    for it in range(args.iters):
        opt.zero_grad(set_to_none=True)

        scales = torch.nn.functional.softplus(scales_raw) + 1e-3
        colors = torch.sigmoid(colors_raw)
        opacities = torch.sigmoid(opacities_raw)

        total = torch.tensor(0.0, device=device)
        for i, tgt in enumerate(targets):
            pred = render_gaussians_torch(
                means,
                scales,
                colors,
                opacities,
                cams[i],
                width=args.width,
                height=args.height,
                background=torch.tensor([0.0, 0.0, 0.0], device=device),
                max_gaussians=max(10000, n),
            )
            total = total + torch.mean(torch.abs(pred - tgt))

        loss = total / len(targets)
        loss.backward()
        opt.step()

        lv = float(loss.detach().cpu())
        loss_log.append(lv)
        if it == 0 or (it + 1) % 25 == 0:
            print(f"iter {it+1:4d}  loss={lv:.6f}")

    np.savez(
        out_dir / "gaussians_fitted.npz",
        means=means.detach().cpu().numpy().astype(np.float32),
        scales=(torch.nn.functional.softplus(scales_raw) + 1e-3).detach().cpu().numpy().astype(np.float32),
        colors=torch.sigmoid(colors_raw).detach().cpu().numpy().astype(np.float32),
        opacities=torch.sigmoid(opacities_raw).detach().cpu().numpy().astype(np.float32),
    )
    (out_dir / "loss.txt").write_text("\n".join(f"{v:.8f}" for v in loss_log), encoding="utf-8")

    with torch.no_grad():
        pred0 = render_gaussians_torch(
            means,
            torch.nn.functional.softplus(scales_raw) + 1e-3,
            torch.sigmoid(colors_raw),
            torch.sigmoid(opacities_raw),
            cams[0],
            width=args.width,
            height=args.height,
            background=torch.tensor([0.0, 0.0, 0.0], device=device),
            max_gaussians=max(10000, n),
        )
        pred_u8 = (pred0.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        Image.fromarray(pred_u8, mode="RGB").save(out_dir / "preview_view0.png")

    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
