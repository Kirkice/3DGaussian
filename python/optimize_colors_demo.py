import argparse

import numpy as np
import torch

from torch_renderer import Camera, look_at, perspective, render_gaussians_torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="../assets/manbo.obj")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import gaussian_renderer as gr

    g = gr.load_obj_as_gaussians(args.obj, default_scale=0.02, default_opacity=0.9)
    means = torch.from_numpy(np.asarray(g["means"], dtype=np.float32)).to(device)
    scales = torch.from_numpy(np.asarray(g["scales"], dtype=np.float32)).to(device)
    opacities = torch.from_numpy(np.asarray(g["opacities"], dtype=np.float32)).to(device)

    # Camera
    eye = torch.tensor([0.0, 0.0, 2.5], device=device)
    target = torch.tensor([0.0, 0.0, 0.0], device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    view = look_at(eye, target, up)
    proj = perspective(60.0, args.width / args.height, 0.01, 100.0, device=device)
    cam = Camera(view=view, proj=proj)

    # Create a synthetic target by assigning a fixed random color per gaussian.
    torch.manual_seed(0)
    target_colors = torch.rand((means.shape[0], 3), device=device)

    with torch.no_grad():
        target_img = render_gaussians_torch(
            means, scales, target_colors, opacities, cam,
            width=args.width, height=args.height,
            background=torch.tensor([0.02, 0.02, 0.02], device=device),
        )

    # Optimize colors starting from gray.
    colors = torch.nn.Parameter(torch.full_like(target_colors, 0.5))
    opt = torch.optim.Adam([colors], lr=args.lr)

    for it in range(args.iters):
        opt.zero_grad(set_to_none=True)
        img = render_gaussians_torch(
            means, scales, colors.sigmoid(), opacities, cam,
            width=args.width, height=args.height,
            background=torch.tensor([0.02, 0.02, 0.02], device=device),
        )
        loss = torch.mean((img - target_img) ** 2)
        loss.backward()
        opt.step()

        if (it + 1) % 20 == 0 or it == 0:
            print(f"iter {it+1:4d}  loss={loss.item():.6f}")

    # Save result
    out = (img.detach().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    target_out = (target_img.detach().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)

    from PIL import Image

    Image.fromarray(target_out, mode="RGB").save("target.png")
    Image.fromarray(out, mode="RGB").save("optimized.png")
    print("Wrote target.png and optimized.png")


if __name__ == "__main__":
    main()
