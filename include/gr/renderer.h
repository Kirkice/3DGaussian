#pragma once

#include <cstdint>

#include "gr/gaussian_types.h"

namespace gr {

// CPU reference renderer (supports depth-sorted alpha compositing).
std::vector<std::uint8_t> render_gaussians_cpu(
    const float* means,
    const float* scales,
    const float* colors,
    const float* opacities,
    int n,
    const RenderParams& params);

// Renders Gaussians and returns an RGBA8 image (width*height*4).
// Arrays are expected to be float32 contiguous:
// - means: N*3
// - scales: N*3
// - colors: N*3
// - opacities: N
std::vector<std::uint8_t> render_gaussians_cuda(
    const float* means,
    const float* scales,
    const float* colors,
    const float* opacities,
    int n,
    const RenderParams& params);

// Preferred entry: chooses CUDA when available unless params.force_cpu is set.
std::vector<std::uint8_t> render_gaussians(
    const float* means,
    const float* scales,
    const float* colors,
    const float* opacities,
    int n,
    const RenderParams& params);

}  // namespace gr
