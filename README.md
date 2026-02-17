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

Advanced options:

```bash
python python/fit_multiview_stub.py \
	--targets_dir assets/scene_tex \
	--camera_npz data/cameras.npz \
	--use_sh \
	--densify_interval 80 --prune_interval 80 \
	--silhouette_weight 0.2 --masks_dir data/masks \
	--depth_weight 0.05 --depth_dir data/depth \
	--num_gaussians 1200 --max_gaussians 3000 \
	--iters 600 --width 256 --height 256
```

`camera_npz` format:

- `view`: `(V, 4, 4)` float32
- `proj`: `(V, 4, 4)` float32

Outputs are written to `outputs/fit_multiview_stub`:

- `gaussians_fitted.npz`
- `loss.txt`
- `preview_view0.png`

## View 3D Gaussian model

Interactive viewer:

```bash
python python/view_gaussians.py --npz outputs/fit_multiview_stub/gaussians_fitted.npz
```

Save a rendered snapshot:

```bash
python python/view_gaussians.py --npz outputs/fit_multiview_stub/gaussians_fitted.npz --save outputs/fit_multiview_stub/gaussians_view.png
```

## Native realtime Gaussian viewer (C++)

Build:

```bash
cmake -S . -B build_native -DGR_ENABLE_CUDA=OFF -DGR_BUILD_PYTHON=ON -DGR_BUILD_NATIVE_VIEWER=ON
cmake --build build_native -j --target gaussian_native_viewer
```

Run:

```bash
./build_native/gaussian_native_viewer outputs/fit_scene_tex_m1/gaussians_fitted.npz --width 960 --height 540
```

Controls: left mouse drag orbit, mouse wheel zoom, `R` reset, `H` toggle HUD.

## Notes

- This is a fitting scaffold for the upcoming photo/video multiview pipeline.
- For video input, extract frames to PNGs first, then pass the frame directory as `--targets_dir`.
