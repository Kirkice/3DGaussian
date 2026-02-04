import argparse
import os
import sys

import numpy as np
from PIL import Image


def perspective(fovy_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    f = 1.0 / np.tan(np.deg2rad(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)
    u = up / (np.linalg.norm(up) + 1e-8)
    s = np.cross(f, u)
    s = s / (np.linalg.norm(s) + 1e-8)
    u2 = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u2
    m[2, :3] = -f
    t = np.eye(4, dtype=np.float32)
    t[:3, 3] = -eye
    return m @ t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--out", default="out.png")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--scale", type=float, default=0.01)
    ap.add_argument("--opacity", type=float, default=0.8)
    args = ap.parse_args()

    try:
        import gaussian_renderer as gr
    except Exception as e:
        print("Failed to import gaussian_renderer. Build the CMake project first.")
        raise

    g = gr.load_obj_as_gaussians(args.obj, default_scale=args.scale, default_opacity=args.opacity)

    eye = np.array([0.0, 0.0, 2.5], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    view = look_at(eye, target, up)
    proj = perspective(60.0, args.width / args.height, 0.01, 100.0)

    rgba = gr.render_gaussians(
        g["means"], g["scales"], g["colors"], g["opacities"],
        width=args.width, height=args.height,
        view=view, proj=proj,
        background=np.array([0.02, 0.02, 0.02], dtype=np.float32),
    )

    img = Image.fromarray(rgba, mode="RGBA")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    img.save(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
