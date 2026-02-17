from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    required = ["means", "scales", "colors", "opacities"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")

    means = np.asarray(data["means"], dtype=np.float32)
    scales = np.asarray(data["scales"], dtype=np.float32)
    colors = np.asarray(data["colors"], dtype=np.float32)
    opacities = np.asarray(data["opacities"], dtype=np.float32)
    return means, scales, colors, opacities


def _equal_axes(ax, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.5)
    radius = max(radius, 1e-3)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to gaussians_fitted.npz")
    ap.add_argument("--max_points", type=int, default=50000)
    ap.add_argument("--point_size", type=float, default=8.0)
    ap.add_argument("--alpha_scale", type=float, default=1.0)
    ap.add_argument("--save", default="", help="Optional output png path; if set, save figure and exit")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    means, scales, colors, opacities = _load_npz(npz_path)

    n = means.shape[0]
    if n == 0:
        raise ValueError("No gaussians in model")

    if n > args.max_points:
        idx = np.linspace(0, n - 1, args.max_points, dtype=np.int64)
        means = means[idx]
        scales = scales[idx]
        colors = colors[idx]
        opacities = opacities[idx]

    colors = np.clip(colors, 0.0, 1.0)
    alpha = np.clip(opacities * args.alpha_scale, 0.05, 1.0)
    rgba = np.concatenate([colors, alpha[:, None]], axis=1)

    base = np.mean(np.abs(scales), axis=1)
    base = np.clip(base, 1e-4, None)
    base = base / np.percentile(base, 95)
    marker_size = args.point_size * np.clip(base, 0.1, 3.0)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(means[:, 0], means[:, 1], means[:, 2], s=marker_size, c=rgba, depthshade=False)
    ax.set_title(f"Gaussian Model: {npz_path.name}  (N={means.shape[0]})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _equal_axes(ax, means)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"Saved: {out}")
        return

    print("Close the window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
