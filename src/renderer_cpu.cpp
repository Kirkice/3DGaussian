#include "gr/renderer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace gr {

static inline void mat4_mul_vec4_rowmajor(const float m[16], const float v[4], float out[4]) {
  out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3] * v[3];
  out[1] = m[4] * v[0] + m[5] * v[1] + m[6] * v[2] + m[7] * v[3];
  out[2] = m[8] * v[0] + m[9] * v[1] + m[10] * v[2] + m[11] * v[3];
  out[3] = m[12] * v[0] + m[13] * v[1] + m[14] * v[2] + m[15] * v[3];
}

static inline float clamp01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

std::vector<std::uint8_t> render_gaussians_cuda(
    const float* means,
    const float* scales,
    const float* colors,
    const float* opacities,
    int n,
    const RenderParams& params) {
  const int width = params.width;
  const int height = params.height;
  const int pixels = width * height;

  static thread_local std::vector<float> accum;
  static thread_local std::vector<float> accum_a;
  static thread_local std::vector<std::uint8_t> out;

  if (params.enable_depth_sort != 0) {
    accum.assign(static_cast<size_t>(pixels) * 3, 0.0f);
    accum_a.assign(static_cast<size_t>(pixels), 0.0f);
  } else {
    accum.assign(static_cast<size_t>(pixels) * 4, 0.0f);
  }

  if (n > 0 && params.enable_depth_sort == 0) {
    const float fx = std::fabs(params.proj[0]);
    const float fy = std::fabs(params.proj[5]);

    for (int i = 0; i < n; ++i) {
      const float x = means[i * 3 + 0];
      const float y = means[i * 3 + 1];
      const float z = means[i * 3 + 2];

      const float sx = scales[i * 3 + 0];
      const float sy = scales[i * 3 + 1];

      const float cr = colors[i * 3 + 0];
      const float cg = colors[i * 3 + 1];
      const float cb = colors[i * 3 + 2];
      const float opacity = opacities[i];

      const float p_obj[4] = {x, y, z, 1.0f};
      float p_cam[4];
      mat4_mul_vec4_rowmajor(params.view, p_obj, p_cam);
      const float z_abs = std::fabs(p_cam[2]) + 1e-6f;

      float p_clip[4];
      mat4_mul_vec4_rowmajor(params.proj, p_cam, p_clip);
      if (p_clip[3] == 0.0f) continue;

      const float inv_w = 1.0f / p_clip[3];
      const float ndc_x = p_clip[0] * inv_w;
      const float ndc_y = p_clip[1] * inv_w;
      const float ndc_z = p_clip[2] * inv_w;
      if (ndc_z < -1.0f || ndc_z > 1.0f) continue;

      const float px = (ndc_x * 0.5f + 0.5f) * (width - 1);
      const float py = (1.0f - (ndc_y * 0.5f + 0.5f)) * (height - 1);

      float sigma_x = sx * 0.5f * width * fx / z_abs;
      float sigma_y = sy * 0.5f * height * fy / z_abs;
      sigma_x = std::max(sigma_x, 1.0f);
      sigma_y = std::max(sigma_y, 1.0f);

      const float rad_x = 3.0f * sigma_x;
      const float rad_y = 3.0f * sigma_y;

      const int xmin = std::max(0, static_cast<int>(std::floor(px - rad_x)));
      const int xmax = std::min(width - 1, static_cast<int>(std::ceil(px + rad_x)));
      const int ymin = std::max(0, static_cast<int>(std::floor(py - rad_y)));
      const int ymax = std::min(height - 1, static_cast<int>(std::ceil(py + rad_y)));

      const float inv_sx2 = 1.0f / (sigma_x * sigma_x);
      const float inv_sy2 = 1.0f / (sigma_y * sigma_y);

      for (int yy = ymin; yy <= ymax; ++yy) {
        for (int xx = xmin; xx <= xmax; ++xx) {
          const float dx = (static_cast<float>(xx) + 0.5f) - px;
          const float dy = (static_cast<float>(yy) + 0.5f) - py;
          const float e = -0.5f * (dx * dx * inv_sx2 + dy * dy * inv_sy2);
          const float w = opacity * std::exp(e);
          if (w < 1e-5f) continue;

          const int idx = (yy * width + xx) * 4;
          accum[idx + 0] += w * cr;
          accum[idx + 1] += w * cg;
          accum[idx + 2] += w * cb;
          accum[idx + 3] += w;
        }
      }
    }
  }

  if (n > 0 && params.enable_depth_sort != 0) {
    const float fx = std::fabs(params.proj[0]);
    const float fy = std::fabs(params.proj[5]);

    // Global depth sort (camera-space z). For this view matrix convention, points in front typically have negative z;
    // larger z (less negative) is closer to the camera.
    static thread_local std::vector<int> order;
    static thread_local std::vector<float> zcam;
    order.resize(static_cast<size_t>(n));
    zcam.resize(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);

    for (int i = 0; i < n; ++i) {
      const float p_obj[4] = {means[i * 3 + 0], means[i * 3 + 1], means[i * 3 + 2], 1.0f};
      float p_cam[4];
      mat4_mul_vec4_rowmajor(params.view, p_obj, p_cam);
      zcam[static_cast<size_t>(i)] = p_cam[2];
    }

    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return zcam[static_cast<size_t>(a)] > zcam[static_cast<size_t>(b)];
    });

    for (int oi = 0; oi < n; ++oi) {
      const int i = order[static_cast<size_t>(oi)];
      const float x = means[i * 3 + 0];
      const float y = means[i * 3 + 1];
      const float z = means[i * 3 + 2];

      const float sx = scales[i * 3 + 0];
      const float sy = scales[i * 3 + 1];

      const float cr = colors[i * 3 + 0];
      const float cg = colors[i * 3 + 1];
      const float cb = colors[i * 3 + 2];
      const float opacity = opacities[i];

      const float p_obj[4] = {x, y, z, 1.0f};
      float p_cam[4];
      mat4_mul_vec4_rowmajor(params.view, p_obj, p_cam);
      const float z_abs = std::fabs(p_cam[2]) + 1e-6f;

      float p_clip[4];
      mat4_mul_vec4_rowmajor(params.proj, p_cam, p_clip);
      if (p_clip[3] == 0.0f) continue;

      const float inv_w = 1.0f / p_clip[3];
      const float ndc_x = p_clip[0] * inv_w;
      const float ndc_y = p_clip[1] * inv_w;
      const float ndc_z = p_clip[2] * inv_w;
      if (ndc_z < -1.0f || ndc_z > 1.0f) continue;

      const float px = (ndc_x * 0.5f + 0.5f) * (width - 1);
      const float py = (1.0f - (ndc_y * 0.5f + 0.5f)) * (height - 1);

      float sigma_x = sx * 0.5f * width * fx / z_abs;
      float sigma_y = sy * 0.5f * height * fy / z_abs;
      sigma_x = std::max(sigma_x, 1.0f);
      sigma_y = std::max(sigma_y, 1.0f);

      const float rad_x = 3.0f * sigma_x;
      const float rad_y = 3.0f * sigma_y;

      const int xmin = std::max(0, static_cast<int>(std::floor(px - rad_x)));
      const int xmax = std::min(width - 1, static_cast<int>(std::ceil(px + rad_x)));
      const int ymin = std::max(0, static_cast<int>(std::floor(py - rad_y)));
      const int ymax = std::min(height - 1, static_cast<int>(std::ceil(py + rad_y)));

      const float inv_sx2 = 1.0f / (sigma_x * sigma_x);
      const float inv_sy2 = 1.0f / (sigma_y * sigma_y);

      for (int yy = ymin; yy <= ymax; ++yy) {
        for (int xx = xmin; xx <= xmax; ++xx) {
          const float dx = (static_cast<float>(xx) + 0.5f) - px;
          const float dy = (static_cast<float>(yy) + 0.5f) - py;
          const float e = -0.5f * (dx * dx * inv_sx2 + dy * dy * inv_sy2);
          float a = opacity * std::exp(e);
          if (a < 1e-5f) continue;
          a = clamp01(a);

          const int pidx = yy * width + xx;
          const float one_minus = 1.0f - accum_a[static_cast<size_t>(pidx)];
          const float contrib = one_minus * a;
          if (contrib <= 0.0f) continue;

          accum[static_cast<size_t>(pidx) * 3 + 0] += contrib * cr;
          accum[static_cast<size_t>(pidx) * 3 + 1] += contrib * cg;
          accum[static_cast<size_t>(pidx) * 3 + 2] += contrib * cb;
          accum_a[static_cast<size_t>(pidx)] += contrib;
        }
      }
    }
  }

  out.resize(static_cast<size_t>(pixels) * 4);
  if (params.enable_depth_sort == 0) {
    for (int i = 0; i < pixels; ++i) {
      const float a_r = accum[i * 4 + 0];
      const float a_g = accum[i * 4 + 1];
      const float a_b = accum[i * 4 + 2];
      const float a_w = accum[i * 4 + 3];

      const float denom = 1.0f + a_w;
      float r = (params.background[0] + a_r) / denom;
      float g = (params.background[1] + a_g) / denom;
      float b = (params.background[2] + a_b) / denom;

      r = std::min(std::max(r, 0.0f), 1.0f);
      g = std::min(std::max(g, 0.0f), 1.0f);
      b = std::min(std::max(b, 0.0f), 1.0f);

      out[i * 4 + 0] = static_cast<std::uint8_t>(r * 255.0f + 0.5f);
      out[i * 4 + 1] = static_cast<std::uint8_t>(g * 255.0f + 0.5f);
      out[i * 4 + 2] = static_cast<std::uint8_t>(b * 255.0f + 0.5f);
      out[i * 4 + 3] = 255;
    }
  } else {
    for (int i = 0; i < pixels; ++i) {
      const float a = std::min(std::max(accum_a[static_cast<size_t>(i)], 0.0f), 1.0f);
      float r = accum[static_cast<size_t>(i) * 3 + 0] + (1.0f - a) * params.background[0];
      float g = accum[static_cast<size_t>(i) * 3 + 1] + (1.0f - a) * params.background[1];
      float b = accum[static_cast<size_t>(i) * 3 + 2] + (1.0f - a) * params.background[2];

      r = std::min(std::max(r, 0.0f), 1.0f);
      g = std::min(std::max(g, 0.0f), 1.0f);
      b = std::min(std::max(b, 0.0f), 1.0f);

      out[i * 4 + 0] = static_cast<std::uint8_t>(r * 255.0f + 0.5f);
      out[i * 4 + 1] = static_cast<std::uint8_t>(g * 255.0f + 0.5f);
      out[i * 4 + 2] = static_cast<std::uint8_t>(b * 255.0f + 0.5f);
      out[i * 4 + 3] = 255;
    }
  }

  return out;
}

}  // namespace gr
