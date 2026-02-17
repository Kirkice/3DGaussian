# 3D Gaussian Multiview Fitting (Stub)

This project now keeps only the **new multiview fitting workflow**.
Legacy OBJ/MTL/texture conversion and old CLI/Viewer paths were removed.

## What remains

- Pure PyTorch reference renderer utilities: [python/torch_renderer.py](python/torch_renderer.py)
- Device selection policy: [python/device_utils.py](python/device_utils.py)
- New fitting entrypoint: [python/fit_multiview_stub.py](python/fit_multiview_stub.py)

## Device policy

Default runtime device is selected as:

- Windows + CUDA available -> `cuda`
- Otherwise -> `mps` (if available) else `cpu`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r python/requirements_torch.txt
```

## Run multiview fitting stub

Prepare a folder containing target PNG views, then run:

```bash
python python/fit_multiview_stub.py --targets_dir assets/targetTexture --iters 150 --width 128 --height 128
```

Outputs are written to `outputs/fit_multiview_stub`:

- `gaussians_fitted.npz`
- `loss.txt`
- `preview_view0.png`

## Notes

- This is a fitting scaffold for the upcoming photo/video multiview pipeline.
- For video input, extract frames to PNGs first, then pass the frame directory as `--targets_dir`.
