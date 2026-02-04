# 3D Gaussian Renderer (OBJ -> Gaussians)

A minimal baseline project that:

- Loads an `.obj` mesh and converts vertices to simple 3D Gaussians
- Renders those Gaussians to a 2D image using a CUDA splat renderer
- Exposes everything as a Python module via `pybind11`

This is **not** a full "3D Gaussian Splatting" implementation yet (no learned SH, no tile binning, no EWA covariance, etc.). It is a clean starting point you can extend.

## Prerequisites (Windows)

- Visual Studio 2022 (MSVC toolchain)
- CUDA Toolkit installed and working with VS
- CMake 3.24+
- Python 3.9+ (recommended 3.10/3.11)

## Build

From a Developer PowerShell (or VS x64 Native Tools shell):

```powershell
cd f:\Project\CppProject\3D
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

If CMake can't find Python on Windows, point it explicitly:

```powershell
cmake .. -G "Visual Studio 17 2022" -A x64 -DPython_ROOT_DIR="C:\\Path\\To\\Python"
```

If CUDA is not available on your machine, you can still build and run the demo using the CPU fallback renderer:

```powershell
cmake .. -G "Visual Studio 17 2022" -A x64 -DGR_ENABLE_CUDA=OFF
cmake --build . --config Release
```

If you don't need Python bindings (or Python isn't installed), you can still build the C++ CLI:

```powershell
cmake .. -G "Visual Studio 17 2022" -A x64 -DGR_BUILD_PYTHON=OFF
cmake --build . --config Release
```

Run the CLI:

```powershell
build\Release\gaussian_render_cli.exe path\to\model.obj out.ppm 960 540
```

## Realtime viewer (EXE)

Build (CPU fallback example):

```powershell
cmake -S . -B build_viewer -G "Visual Studio 17 2022" -A x64 -DGR_ENABLE_CUDA=OFF -DGR_BUILD_PYTHON=OFF -DGR_BUILD_VIEWER=ON
cmake --build build_viewer --config Release
```

Build with CUDA (GPU renderer):

> Note: PyTorch CUDA wheels do **not** include `nvcc`. To compile this project's `.cu` files you must install the **NVIDIA CUDA Toolkit** (so `nvcc` is available).

```powershell
cmake -S . -B build_viewer_cuda -G "Visual Studio 17 2022" -A x64 -DGR_ENABLE_CUDA=ON -DGR_BUILD_PYTHON=OFF -DGR_BUILD_VIEWER=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build_viewer_cuda --config Release
```

Run:

```powershell
build_viewer\Release\gaussian_viewer.exe assets\manbo.obj
```

Controls: left mouse drag = orbit, mouse wheel = zoom, `R` = reset.

Quick sanity check with the included cube:

```powershell
build\Release\gaussian_render_cli.exe assets\manbo.obj out.ppm 960 540
```

The build outputs a Python extension module named `gaussian_renderer*.pyd` under the build folder.

## Run demo

```powershell
cd f:\Project\CppProject\3D
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r python\requirements.txt

# Point this at your obj file
python python\demo_render_obj.py --obj path\to\model.obj --out out.png
```

## Differentiable (PyTorch) experiment loop

For learning/optimization experiments, this repo includes a **pure PyTorch reference renderer** (slow but differentiable):

- [python/torch_renderer.py](python/torch_renderer.py)

It is intended for small resolutions (e.g. 128×128) and modest Gaussian counts.

### Install

```powershell
pip install -r python\requirements_torch.txt
```

Install `torch` per the official instructions (CPU or CUDA build).

### Run optimization demo

This demo optimizes per-Gaussian colors to match a synthetic target:

```powershell
python python\optimize_colors_demo.py --obj assets\manbo.obj --iters 200 --width 128 --height 128
```

Outputs: `target.png`, `optimized.png`.

## Notes

- Conversion: currently 1 vertex -> 1 Gaussian, with constant scale and opacity.
- Renderer: naive per-Gaussian splat into an accumulation buffer (good enough to verify the pipeline).

Next steps typically include: sampling more points on faces, per-Gaussian anisotropy + rotation, proper alpha compositing, tile-based binning, and SH shading.
