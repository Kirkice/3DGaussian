#include "gr/renderer.h"

namespace gr {

std::vector<std::uint8_t> render_gaussians(
    const float* means,
    const float* scales,
    const float* colors,
    const float* opacities,
    int n,
    const RenderParams& params) {
  if (params.force_cpu != 0) {
    return render_gaussians_cpu(means, scales, colors, opacities, n, params);
  }

#if GR_CUDA_ENABLED != 0
  return render_gaussians_cuda(means, scales, colors, opacities, n, params);
#else
  return render_gaussians_cpu(means, scales, colors, opacities, n, params);
#endif
}

}  // namespace gr
