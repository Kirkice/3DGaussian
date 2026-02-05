import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from torch_renderer import Camera, look_at, perspective, render_gaussians_torch


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    # x in [0,1]
    return torch.where(x <= 0.04045, x / 12.92, torch.pow((x + 0.055) / 1.055, 2.4))


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * torch.pow(x, 1.0 / 2.4) - 0.055)


def _parse_rgb_triplet(s: str) -> torch.Tensor:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected R,G,B")
    try:
        rgb = [float(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError("expected float R,G,B") from e
    return torch.tensor(rgb, dtype=torch.float32)


def _load_image_rgb(path: Path, width: int, height: int) -> torch.Tensor:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.width != width or img.height != height:
        img = img.resize((width, height), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)  # (H,W,3)
    return t


def _estimate_background(img_rgb: torch.Tensor, patch: int = 16) -> torch.Tensor:
    # Use small corner patches; robust median.
    h, w, _ = img_rgb.shape
    p = max(1, min(patch, h // 4, w // 4))

    corners = torch.cat(
        [
            img_rgb[0:p, 0:p].reshape(-1, 3),
            img_rgb[0:p, w - p : w].reshape(-1, 3),
            img_rgb[h - p : h, 0:p].reshape(-1, 3),
            img_rgb[h - p : h, w - p : w].reshape(-1, 3),
        ],
        dim=0,
    )
    return corners.median(dim=0).values


def _make_orbit_cameras(
    *,
    width: int,
    height: int,
    fovy_deg: float,
    radius: float,
    pitch_rad: float,
    yaw_offset_deg: float,
    num_views: int,
    device: torch.device,
) -> List[Camera]:
    cams: List[Camera] = []
    proj = perspective(fovy_deg, width / height, 0.01, 100.0, device=device)

    target = torch.tensor([0.0, 0.0, 0.0], device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)

    for i in range(num_views):
        yaw_rad = (yaw_offset_deg + (360.0 * i / num_views)) * math.pi / 180.0
        eye = torch.tensor(
            [
                radius * math.cos(pitch_rad) * math.sin(yaw_rad),
                radius * math.sin(pitch_rad),
                radius * math.cos(pitch_rad) * math.cos(yaw_rad),
            ],
            device=device,
            dtype=torch.float32,
        )
        view = look_at(eye, target, up)
        cams.append(Camera(view=view, proj=proj))

    return cams


def _logit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


@dataclass
class TargetView:
    path: Path
    img_linear: torch.Tensor  # (H,W,3) float32
    background_linear: torch.Tensor  # (3,)
    mask: torch.Tensor  # (H,W,1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="../assets/manbo.obj")
    ap.add_argument("--targets_dir", default="../assets/targetTexture")
    ap.add_argument("--out_dir", default="../outputs/opt_colors_multiview")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)

    ap.add_argument("--default_scale", type=float, default=0.02)
    ap.add_argument("--default_opacity", type=float, default=0.9)
    ap.add_argument(
        "--num_surface_samples",
        type=int,
        default=4000,
        help="Gaussian count for surface sampling. Keep <= 10000 for the torch reference renderer.",
    )

    ap.add_argument("--fovy", type=float, default=60.0)
    ap.add_argument("--radius", type=float, default=2.5)
    ap.add_argument("--pitch", type=float, default=0.2, help="pitch in radians (matches viewer default)")
    ap.add_argument("--yaw_offset_deg", type=float, default=0.0)

    ap.add_argument(
        "--background",
        type=_parse_rgb_triplet,
        default=None,
        help="Optional override background as R,G,B in 0..1. If omitted, estimated from target corners per view.",
    )
    ap.add_argument(
        "--srgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assume target PNGs are sRGB and compare in linear space.",
    )
    ap.add_argument(
        "--mask_eps",
        type=float,
        default=0.06,
        help="Mask threshold in linear space for distinguishing foreground from background.",
    )
    ap.add_argument(
        "--mask_soft",
        type=float,
        default=0.02,
        help="Softness for mask ramp; 0 disables soft mask.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    targets_dir = Path(args.targets_dir)
    if not targets_dir.exists():
        raise FileNotFoundError(f"targets_dir not found: {targets_dir}")

    image_paths = sorted(targets_dir.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if len(image_paths) == 0:
        raise FileNotFoundError(f"no .png found under: {targets_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load targets
    target_views: List[TargetView] = []
    for p in image_paths:
        img = _load_image_rgb(p, args.width, args.height).to(dtype=torch.float32)

        bg = args.background if args.background is not None else _estimate_background(img)

        img_lin = _srgb_to_linear(img) if args.srgb else img
        bg_lin = _srgb_to_linear(bg) if args.srgb else bg

        diff = (img_lin - bg_lin.view(1, 1, 3)).abs().mean(dim=2, keepdim=True)
        if args.mask_soft > 0.0:
            mask = ((diff - args.mask_eps) / args.mask_soft).clamp(0.0, 1.0)
        else:
            mask = (diff > args.mask_eps).to(dtype=torch.float32)
        mask = mask.detach()

        target_views.append(TargetView(path=p, img_linear=img_lin, background_linear=bg_lin, mask=mask))

    num_views = len(target_views)
    print(f"Loaded {num_views} target views from {targets_dir}")

    # Create cameras (simple orbit). This matches the viewer's camera orbit convention.
    cams = _make_orbit_cameras(
        width=args.width,
        height=args.height,
        fovy_deg=args.fovy,
        radius=args.radius,
        pitch_rad=args.pitch,
        yaw_offset_deg=args.yaw_offset_deg,
        num_views=num_views,
        device=device,
    )

    # Load gaussians from OBJ
    import gaussian_renderer as gr

    g = gr.load_obj_as_gaussians(
        args.obj,
        default_scale=args.default_scale,
        default_opacity=args.default_opacity,
        num_surface_samples=args.num_surface_samples,
    )

    means = torch.from_numpy(np.asarray(g["means"], dtype=np.float32)).to(device)
    scales = torch.from_numpy(np.asarray(g["scales"], dtype=np.float32)).to(device)
    opacities = torch.from_numpy(np.asarray(g["opacities"], dtype=np.float32)).to(device)

    n = means.shape[0]
    if n > 10000:
        raise ValueError(
            f"N={n} too large for torch reference renderer. "
            "Lower --num_surface_samples (recommended <= 10000)."
        )
    print(f"Gaussians: N={n}")

    # Trainable colors (logits -> sigmoid)
    init_colors = torch.full((n, 3), 0.85, dtype=torch.float32, device=device)
    colors_logits = torch.nn.Parameter(_logit(init_colors))
    opt = torch.optim.Adam([colors_logits], lr=args.lr)

    # Move targets to device once.
    for tv in target_views:
        tv.img_linear = tv.img_linear.to(device)
        tv.background_linear = tv.background_linear.to(device)
        tv.mask = tv.mask.to(device)

    def render_view(view_idx: int, colors01: torch.Tensor) -> torch.Tensor:
        tv = target_views[view_idx]
        return render_gaussians_torch(
            means,
            scales,
            colors01,
            opacities,
            cams[view_idx],
            width=args.width,
            height=args.height,
            background=tv.background_linear,
            max_gaussians=10000,
        )

    # Training loop
    loss_history: List[float] = []
    for it in range(args.iters):
        opt.zero_grad(set_to_none=True)

        colors01 = colors_logits.sigmoid()
        total = torch.tensor(0.0, device=device)
        denom = torch.tensor(0.0, device=device)

        for vi in range(num_views):
            pred = render_view(vi, colors01)
            tv = target_views[vi]
            mask = tv.mask
            # L1 loss with foreground emphasis; keep a tiny weight everywhere for stability.
            w = 0.05 + 0.95 * mask
            err = (pred - tv.img_linear).abs() * w
            total = total + err.mean()
            denom = denom + 1.0

        loss = total / denom
        loss.backward()
        opt.step()

        loss_val = float(loss.detach().cpu().item())
        loss_history.append(loss_val)
        if (it + 1) % 25 == 0 or it == 0:
            print(f"iter {it + 1:4d}  loss={loss_val:.6f}")

    # Save trained colors + debug renders
    colors_trained = colors_logits.sigmoid().detach()
    np.save(out_dir / "trained_colors.npy", colors_trained.cpu().numpy().astype(np.float32))

    # Also write a C++ friendly binary colors file.
    # Format: 'GRCL' + uint32 N + float32 RGB (N*3).
    colors_bin_path = out_dir / "colors.bin"
    with open(colors_bin_path, "wb") as f:
        f.write(b"GRCL")
        f.write(np.asarray([colors_trained.shape[0]], dtype=np.uint32).tobytes())
        f.write(colors_trained.cpu().numpy().astype(np.float32).tobytes(order="C"))

    np.savez(
        out_dir / "gaussians_trained.npz",
        means=means.detach().cpu().numpy().astype(np.float32),
        scales=scales.detach().cpu().numpy().astype(np.float32),
        colors=colors_trained.cpu().numpy().astype(np.float32),
        opacities=opacities.detach().cpu().numpy().astype(np.float32),
    )

    (out_dir / "loss.txt").write_text("\n".join(f"{v:.8f}" for v in loss_history), encoding="utf-8")

    from PIL import Image

    with torch.no_grad():
        for vi in range(num_views):
            pred_lin = render_view(vi, colors_trained)
            tgt_lin = target_views[vi].img_linear

            pred_srgb = _linear_to_srgb(pred_lin) if args.srgb else pred_lin
            tgt_srgb = _linear_to_srgb(tgt_lin) if args.srgb else tgt_lin

            pred_u8 = (pred_srgb.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            tgt_u8 = (tgt_srgb.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)

            Image.fromarray(tgt_u8, mode="RGB").save(out_dir / f"target_{vi}.png")
            Image.fromarray(pred_u8, mode="RGB").save(out_dir / f"optimized_{vi}.png")

    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
