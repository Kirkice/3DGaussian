from __future__ import annotations

from dataclasses import dataclass
import platform
from typing import Optional, Tuple

import torch


@dataclass
class Camera:
    view: torch.Tensor  # (4,4) float32
    proj: torch.Tensor  # (4,4) float32


def get_default_device() -> torch.device:
    if platform.system() == "Windows" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def perspective(fovy_deg: float, aspect: float, znear: float, zfar: float, device=None) -> torch.Tensor:
    f = 1.0 / torch.tan(torch.tensor(fovy_deg, device=device) * torch.pi / 180.0 * 0.5)
    m = torch.zeros((4, 4), dtype=torch.float32, device=device)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    eye = eye.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)
    up = up.to(dtype=torch.float32)

    f = target - eye
    f = f / (torch.linalg.norm(f) + 1e-8)
    u = up / (torch.linalg.norm(up) + 1e-8)
    s = torch.linalg.cross(f, u)
    s = s / (torch.linalg.norm(s) + 1e-8)
    u2 = torch.linalg.cross(s, f)

    m = torch.eye(4, dtype=torch.float32, device=eye.device)
    m[0, :3] = s
    m[1, :3] = u2
    m[2, :3] = -f

    t = torch.eye(4, dtype=torch.float32, device=eye.device)
    t[:3, 3] = -eye
    return m @ t


def _project(means: torch.Tensor, view: torch.Tensor, proj: torch.Tensor, width: int, height: int):
    # means: (N,3)
    n = means.shape[0]
    ones = torch.ones((n, 1), dtype=means.dtype, device=means.device)
    p_obj = torch.cat([means, ones], dim=1)  # (N,4)

    p_cam = (view @ p_obj.t()).t()  # (N,4)
    p_clip = (proj @ p_cam.t()).t()  # (N,4)

    w = p_clip[:, 3:4]
    w_safe = torch.where(w.abs() < 1e-8, torch.ones_like(w), w)
    ndc = p_clip[:, :3] / w_safe

    # screen coords
    px = (ndc[:, 0] * 0.5 + 0.5) * (width - 1)
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (height - 1)

    # visibility mask (rough)
    valid = (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0) & (w.squeeze(1) != 0.0)
    z_abs = p_cam[:, 2].abs().clamp_min(1e-6)

    return px, py, z_abs, valid


def render_gaussians_torch(
    means: torch.Tensor,      # (N,3) float32
    scales: torch.Tensor,     # (N,3) float32
    colors: torch.Tensor,     # (N,3) float32 (0..1)
    opacities: torch.Tensor,  # (N,)  float32
    camera: Camera,
    width: int,
    height: int,
    background: Optional[torch.Tensor] = None,  # (3,)
    max_gaussians: int = 10000,
    chunk_size: int = 256,
) -> torch.Tensor:
    """A *differentiable* reference renderer in PyTorch.

    It is intentionally simple (O(N*H*W)). Use for experiments/optimization at small sizes.
    For real-time performance you still want the CUDA rasterizer.
    """

    if background is None:
        background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=means.device)
    background = background.to(dtype=torch.float32, device=means.device)

    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must be (N,3)")
    n = means.shape[0]
    if n == 0:
        return torch.zeros((height, width, 3), dtype=torch.float32, device=means.device)
    if n > max_gaussians:
        raise ValueError(f"N={n} too large for torch reference renderer. Increase max_gaussians or downsample.")

    view = camera.view.to(dtype=torch.float32, device=means.device)
    proj = camera.proj.to(dtype=torch.float32, device=means.device)

    px, py, z_abs, valid = _project(means, view, proj, width, height)

    # Approximate screen-space sigma similar to C++ path.
    fx = proj[0, 0].abs()
    fy = proj[1, 1].abs()
    sigma_x = (scales[:, 0].abs() * 0.5 * width * fx / z_abs).clamp_min(1.0)
    sigma_y = (scales[:, 1].abs() * 0.5 * height * fy / z_abs).clamp_min(1.0)

    # Pixel grid (H,W)
    ys = torch.arange(height, device=means.device, dtype=torch.float32) + 0.5
    xs = torch.arange(width, device=means.device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    hw = height * width
    accum_rgb_flat = torch.zeros((hw, 3), dtype=torch.float32, device=means.device)
    accum_w_flat = torch.zeros((hw,), dtype=torch.float32, device=means.device)

    # Chunk gaussians to reduce Python overhead.
    # This is still O(N*H*W), but avoids an N-sized Python loop.
    if chunk_size is None or chunk_size <= 0:
        chunk_size = 1

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        v = valid[start:end]
        if not bool(v.any()):
            continue

        px_c = px[start:end].to(torch.float32)
        py_c = py[start:end].to(torch.float32)
        sx_c = sigma_x[start:end].to(torch.float32)
        sy_c = sigma_y[start:end].to(torch.float32)
        op_c = opacities[start:end].to(torch.float32).clamp_min(0.0)
        col_c = colors[start:end].to(torch.float32)

        dx = grid_x.unsqueeze(0) - px_c.view(-1, 1, 1)
        dy = grid_y.unsqueeze(0) - py_c.view(-1, 1, 1)
        e = -0.5 * ((dx * dx) / (sx_c.view(-1, 1, 1) ** 2) + (dy * dy) / (sy_c.view(-1, 1, 1) ** 2))
        w = op_c.view(-1, 1, 1) * torch.exp(e)
        w = w * v.to(dtype=torch.float32).view(-1, 1, 1)

        w_flat = w.reshape(end - start, hw)  # (C,HW)
        accum_w_flat = accum_w_flat + w_flat.sum(dim=0)
        accum_rgb_flat = accum_rgb_flat + (w_flat.t() @ col_c)  # (HW,3)

    accum_rgb = accum_rgb_flat.view(height, width, 3)
    accum_w = accum_w_flat.view(height, width)
    denom = 1.0 + accum_w
    out = (background.view(1, 1, 3) + accum_rgb) / denom.unsqueeze(-1)
    return out.clamp(0.0, 1.0)
