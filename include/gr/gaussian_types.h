#pragma once

#include <cstdint>
#include <vector>

namespace gr {

struct GaussiansHost {
  // Row-major arrays
  // means: N x 3
  std::vector<float> means;
  // scales: N x 3 (object-space isotropic/anisotropic scale)
  std::vector<float> scales;
  // colors: N x 3 (linear 0..1)
  std::vector<float> colors;
  // opacities: N
  std::vector<float> opacities;

  int count() const {
    return static_cast<int>(opacities.size());
  }
};

struct RenderParams {
  int width = 800;
  int height = 600;

  // 4x4 row-major float matrices
  float view[16] = {0};
  float proj[16] = {0};

  float background[3] = {0.0f, 0.0f, 0.0f};

  // Transparency compositing mode.
  // 0: legacy order-independent weighted average (fast, but wrong occlusion)
  // 1: depth-aware compositing (CPU: global sort; CUDA: depth slicing approximation)
  int enable_depth_sort = 0;
  // Only used when enable_depth_sort=1.
  // CUDA mode uses this many depth slices (higher = better ordering, more memory).
  int depth_slices = 16;
};

}  // namespace gr
