from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from device_utils import get_default_device
from torch_renderer import Camera, look_at, perspective, render_gaussians_torch


def _load_image(path: Path, width: int, height: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _load_gray(path: Path, width: int, height: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _list_target_paths(targets_dir: Path) -> list[Path]:
    paths = sorted([*targets_dir.glob("*.png"), *targets_dir.glob("*.jpg"), *targets_dir.glob("*.jpeg")])
    if not paths:
        raise FileNotFoundError(f"No target images found in {targets_dir} (supported: png/jpg/jpeg)")
    return paths


def _load_targets(paths: list[Path], width: int, height: int, device: torch.device) -> list[torch.Tensor]:
    return [torch.from_numpy(_load_image(p, width, height)).to(device) for p in paths]


def _estimate_masks(targets: list[torch.Tensor], thresh: float) -> list[torch.Tensor]:
    masks: list[torch.Tensor] = []
    for t in targets:
        m = (t.mean(dim=2, keepdim=False) > thresh).to(dtype=torch.float32)
        masks.append(m)
    return masks


def _load_optional_masks(paths: list[Path], masks_dir: Path | None, width: int, height: int, device: torch.device) -> list[torch.Tensor] | None:
    if masks_dir is None:
        return None
    out: list[torch.Tensor] = []
    for p in paths:
        candidate = masks_dir / f"{p.stem}.png"
        if not candidate.exists():
            return None
        out.append(torch.from_numpy(_load_gray(candidate, width, height)).to(device))
    return out


def _load_optional_depth(paths: list[Path], depth_dir: Path | None, width: int, height: int, device: torch.device) -> list[torch.Tensor] | None:
    if depth_dir is None:
        return None
    out: list[torch.Tensor] = []
    for p in paths:
        candidate = depth_dir / f"{p.stem}.png"
        if not candidate.exists():
            return None
        d = torch.from_numpy(_load_gray(candidate, width, height)).to(device)
        out.append(d)
    return out


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


def _load_cameras(camera_npz: Path, expected_views: int, device: torch.device) -> list[Camera]:
    data = np.load(camera_npz)
    if "view" not in data or "proj" not in data:
        raise KeyError("camera npz must contain arrays: view (V,4,4), proj (V,4,4)")

    views = np.asarray(data["view"], dtype=np.float32)
    projs = np.asarray(data["proj"], dtype=np.float32)
    if views.shape[0] != expected_views or projs.shape[0] != expected_views:
        raise ValueError("camera count mismatch with number of target images")

    cams: list[Camera] = []
    for i in range(expected_views):
        cams.append(
            Camera(
                view=torch.from_numpy(views[i]).to(device),
                proj=torch.from_numpy(projs[i]).to(device),
            )
        )
    return cams


def _build_params(
    n: int,
    device: torch.device,
    use_sh: bool,
) -> dict[str, torch.nn.Parameter]:
    means = torch.nn.Parameter((torch.rand((n, 3), device=device) - 0.5) * 1.2)
    scales_raw = torch.nn.Parameter(torch.full((n, 3), -2.2, device=device))
    opacities_raw = torch.nn.Parameter(torch.full((n,), -2.2, device=device))

    params: dict[str, torch.nn.Parameter] = {
        "means": means,
        "scales_raw": scales_raw,
        "opacities_raw": opacities_raw,
    }

    if use_sh:
        sh_raw = torch.nn.Parameter(torch.zeros((n, 4, 3), device=device))
        sh_raw.data[:, 0, :] = 0.1 * torch.rand((n, 3), device=device)
        params["sh_raw"] = sh_raw
    else:
        colors_raw = torch.nn.Parameter(0.1 * torch.rand((n, 3), device=device))
        params["colors_raw"] = colors_raw

    return params


def _densify_and_prune(
    params: dict[str, torch.nn.Parameter],
    max_gaussians: int,
    densify_ratio: float,
    prune_opacity: float,
) -> dict[str, torch.nn.Parameter]:
    with torch.no_grad():
        means = params["means"].detach()
        scales_raw = params["scales_raw"].detach()
        op_raw = params["opacities_raw"].detach()
        op = torch.sigmoid(op_raw)
        scales = torch.nn.functional.softplus(scales_raw) + 1e-3

        keep = op > prune_opacity
        if int(keep.sum()) < 64:
            top_keep = torch.topk(op, k=min(64, op.shape[0]), largest=True).indices
            keep = torch.zeros_like(keep, dtype=torch.bool)
            keep[top_keep] = True

        means = means[keep]
        scales_raw = scales_raw[keep]
        op_raw = op_raw[keep]
        op = torch.sigmoid(op_raw)
        scales = scales[keep]

        n = means.shape[0]
        room = max(0, max_gaussians - n)
        add_n = min(room, max(0, int(n * densify_ratio)))

        if add_n > 0 and n > 0:
            idx = torch.topk(op, k=min(n, add_n), largest=True).indices
            jitter = 0.25 * scales[idx] * torch.randn_like(means[idx])
            means = torch.cat([means, means[idx] + jitter], dim=0)
            scales_raw = torch.cat([scales_raw, scales_raw[idx]], dim=0)
            op_raw = torch.cat([op_raw, op_raw[idx] - 0.1], dim=0)

        new_params: dict[str, torch.nn.Parameter] = {
            "means": torch.nn.Parameter(means),
            "scales_raw": torch.nn.Parameter(scales_raw),
            "opacities_raw": torch.nn.Parameter(op_raw),
        }

        if "sh_raw" in params:
            sh_raw = params["sh_raw"].detach()[keep]
            if add_n > 0 and sh_raw.shape[0] > 0:
                base_n = sh_raw.shape[0]
                idx = torch.topk(torch.sigmoid(op_raw[:base_n]), k=min(base_n, add_n), largest=True).indices
                sh_raw = torch.cat([sh_raw, sh_raw[idx]], dim=0)
            new_params["sh_raw"] = torch.nn.Parameter(sh_raw)
        else:
            colors_raw = params["colors_raw"].detach()[keep]
            if add_n > 0 and colors_raw.shape[0] > 0:
                base_n = colors_raw.shape[0]
                idx = torch.topk(torch.sigmoid(op_raw[:base_n]), k=min(base_n, add_n), largest=True).indices
                colors_raw = torch.cat([colors_raw, colors_raw[idx]], dim=0)
            new_params["colors_raw"] = torch.nn.Parameter(colors_raw)

    return new_params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets_dir", required=True, help="Directory containing target images")
    ap.add_argument("--out_dir", default="outputs/fit_multiview_stub")
    ap.add_argument("--camera_npz", default="", help="Optional camera file with view/proj arrays")
    ap.add_argument("--masks_dir", default="", help="Optional silhouette masks dir (same stem as targets, PNG)")
    ap.add_argument("--depth_dir", default="", help="Optional depth maps dir (same stem as targets, PNG normalized)")

    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--num_gaussians", type=int, default=800)
    ap.add_argument("--max_gaussians", type=int, default=3000)

    ap.add_argument("--use_sh", action="store_true", help="Use SH degree-1 color (N,4,3) instead of RGB")

    ap.add_argument("--densify_interval", type=int, default=80)
    ap.add_argument("--prune_interval", type=int, default=80)
    ap.add_argument("--densify_ratio", type=float, default=0.15)
    ap.add_argument("--prune_opacity", type=float, default=0.05)

    ap.add_argument("--silhouette_weight", type=float, default=0.2)
    ap.add_argument("--mask_thresh", type=float, default=0.06)
    ap.add_argument("--depth_weight", type=float, default=0.05)

    ap.add_argument("--reg_opacity", type=float, default=0.001)
    ap.add_argument("--reg_scale", type=float, default=0.001)

    args = ap.parse_args()

    device = get_default_device()
    print(f"Using device: {device}")

    targets_dir = Path(args.targets_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_paths = _list_target_paths(targets_dir)
    targets = _load_targets(target_paths, args.width, args.height, device)

    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    depth_dir = Path(args.depth_dir) if args.depth_dir else None

    masks = _load_optional_masks(target_paths, masks_dir, args.width, args.height, device)
    if masks is None and args.silhouette_weight > 0.0:
        masks = _estimate_masks(targets, args.mask_thresh)

    depths = _load_optional_depth(target_paths, depth_dir, args.width, args.height, device)

    if args.camera_npz:
        cams = _load_cameras(Path(args.camera_npz), len(targets), device)
        print("Using camera poses from camera_npz")
    else:
        cams = _make_orbit_cameras(len(targets), args.width, args.height, device)
        print("Using fallback orbit cameras (for best quality, provide --camera_npz)")

    params = _build_params(args.num_gaussians, device, args.use_sh)

    def trainable_list() -> list[torch.nn.Parameter]:
        return [v for v in params.values()]

    opt = torch.optim.Adam(trainable_list(), lr=args.lr)
    loss_log: list[float] = []

    for it in range(args.iters):
        opt.zero_grad(set_to_none=True)

        means = params["means"]
        scales = torch.nn.functional.softplus(params["scales_raw"]) + 1e-3
        opacities = torch.sigmoid(params["opacities_raw"])

        if args.use_sh:
            colors_eval = params["sh_raw"]
        else:
            colors_eval = torch.sigmoid(params["colors_raw"])

        total = torch.tensor(0.0, device=device)
        for i, tgt in enumerate(targets):
            pred, alpha, depth = render_gaussians_torch(
                means,
                scales,
                colors_eval,
                opacities,
                cams[i],
                width=args.width,
                height=args.height,
                background=torch.tensor([0.0, 0.0, 0.0], device=device),
                max_gaussians=max(args.max_gaussians, means.shape[0]),
                return_aux=True,
            )

            recon = torch.mean(torch.abs(pred - tgt))
            loss_i = recon

            if masks is not None and args.silhouette_weight > 0.0:
                sil = torch.mean(torch.abs(alpha - masks[i]))
                loss_i = loss_i + args.silhouette_weight * sil

            if depths is not None and args.depth_weight > 0.0:
                d_pred = depth / (depth.max() + 1e-6)
                d_gt = depths[i]
                d_loss = torch.mean(torch.abs(d_pred - d_gt))
                loss_i = loss_i + args.depth_weight * d_loss

            total = total + loss_i

        reg = args.reg_opacity * opacities.mean() + args.reg_scale * scales.mean()
        loss = total / len(targets) + reg

        loss.backward()
        opt.step()

        lv = float(loss.detach().cpu())
        loss_log.append(lv)
        if it == 0 or (it + 1) % 25 == 0:
            print(f"iter {it+1:4d}  loss={lv:.6f}  N={means.shape[0]}")

        if (it + 1) % args.prune_interval == 0 or (it + 1) % args.densify_interval == 0:
            params = _densify_and_prune(
                params,
                max_gaussians=args.max_gaussians,
                densify_ratio=args.densify_ratio if (it + 1) % args.densify_interval == 0 else 0.0,
                prune_opacity=args.prune_opacity,
            )
            opt = torch.optim.Adam(trainable_list(), lr=args.lr)

    means = params["means"]
    scales = torch.nn.functional.softplus(params["scales_raw"]) + 1e-3
    opacities = torch.sigmoid(params["opacities_raw"])

    if args.use_sh:
        sh = params["sh_raw"]
        colors = sh[:, 0, :].clamp(0.0, 1.0)
    else:
        sh = None
        colors = torch.sigmoid(params["colors_raw"]) 

    if sh is not None:
        np.savez(
            out_dir / "gaussians_fitted.npz",
            means=means.detach().cpu().numpy().astype(np.float32),
            scales=scales.detach().cpu().numpy().astype(np.float32),
            colors=colors.detach().cpu().numpy().astype(np.float32),
            opacities=opacities.detach().cpu().numpy().astype(np.float32),
            sh_coeffs=sh.detach().cpu().numpy().astype(np.float32),
        )
    else:
        np.savez(
            out_dir / "gaussians_fitted.npz",
            means=means.detach().cpu().numpy().astype(np.float32),
            scales=scales.detach().cpu().numpy().astype(np.float32),
            colors=colors.detach().cpu().numpy().astype(np.float32),
            opacities=opacities.detach().cpu().numpy().astype(np.float32),
        )
    (out_dir / "loss.txt").write_text("\n".join(f"{v:.8f}" for v in loss_log), encoding="utf-8")

    with torch.no_grad():
        if args.use_sh:
            assert sh is not None
            preview_colors: torch.Tensor = sh
        else:
            preview_colors = colors

        pred0 = cast(
            torch.Tensor,
            render_gaussians_torch(
            means,
            scales,
            preview_colors,
            opacities,
            cams[0],
            width=args.width,
            height=args.height,
            background=torch.tensor([0.0, 0.0, 0.0], device=device),
            max_gaussians=max(args.max_gaussians, means.shape[0]),
            return_aux=False,
            ),
        )
        pred_u8 = (pred0.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        Image.fromarray(pred_u8, mode="RGB").save(out_dir / "preview_view0.png")

    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
