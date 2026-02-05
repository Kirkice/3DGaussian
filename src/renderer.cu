#include "gr/renderer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "gr/cuda_utils.cuh"

namespace gr {

static __device__ __forceinline__ float clamp01(float x) {
  return fminf(fmaxf(x, 0.0f), 1.0f);
}

static __device__ __forceinline__ float4 mat4_mul_vec4_rowmajor(const float m[16], const float4 v) {
  float4 r;
  r.x = m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3] * v.w;
  r.y = m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7] * v.w;
  r.z = m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11] * v.w;
  r.w = m[12] * v.x + m[13] * v.y + m[14] * v.z + m[15] * v.w;
  return r;
}

__global__ void splat_kernel(
    const float* means,
    const float* scales,
    const float* colors,
    const float* opacities,
    int n,
    int width,
    int height,
    const float* view,
    const float* proj,
    float4* accum) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float x = means[i * 3 + 0];
  const float y = means[i * 3 + 1];
  const float z = means[i * 3 + 2];

  const float sx = scales[i * 3 + 0];
  const float sy = scales[i * 3 + 1];
  // const float sz = scales[i * 3 + 2];

  const float cr = colors[i * 3 + 0];
  const float cg = colors[i * 3 + 1];
  const float cb = colors[i * 3 + 2];
  const float opacity = opacities[i];

  const float4 p_obj = make_float4(x, y, z, 1.0f);
  const float4 p_cam = mat4_mul_vec4_rowmajor(view, p_obj);
  const float z_abs = fabsf(p_cam.z) + 1e-6f;

  const float4 p_clip = mat4_mul_vec4_rowmajor(proj, p_cam);
  if (p_clip.w == 0.0f) return;

  const float inv_w = 1.0f / p_clip.w;
  const float ndc_x = p_clip.x * inv_w;
  const float ndc_y = p_clip.y * inv_w;
  const float ndc_z = p_clip.z * inv_w;
  if (ndc_z < -1.0f || ndc_z > 1.0f) return;

  const float px = (ndc_x * 0.5f + 0.5f) * (width - 1);
  const float py = (1.0f - (ndc_y * 0.5f + 0.5f)) * (height - 1);

  // Approximate screen-space sigma using projection scaling terms.
  const float fx = fabsf(proj[0]);   // proj[0,0]
  const float fy = fabsf(proj[5]);   // proj[1,1]
  float sigma_x = sx * 0.5f * width * fx / z_abs;
  float sigma_y = sy * 0.5f * height * fy / z_abs;
  sigma_x = fmaxf(sigma_x, 1.0f);
  sigma_y = fmaxf(sigma_y, 1.0f);

  const float rad_x = 3.0f * sigma_x;
  const float rad_y = 3.0f * sigma_y;

  const int xmin = max(0, static_cast<int>(floorf(px - rad_x)));
  const int xmax = min(width - 1, static_cast<int>(ceilf(px + rad_x)));
  const int ymin = max(0, static_cast<int>(floorf(py - rad_y)));
  const int ymax = min(height - 1, static_cast<int>(ceilf(py + rad_y)));

  const float inv_sx2 = 1.0f / (sigma_x * sigma_x);
  const float inv_sy2 = 1.0f / (sigma_y * sigma_y);

  for (int yy = ymin; yy <= ymax; ++yy) {
    for (int xx = xmin; xx <= xmax; ++xx) {
      const float dx = (static_cast<float>(xx) + 0.5f) - px;
      const float dy = (static_cast<float>(yy) + 0.5f) - py;
      const float e = -0.5f * (dx * dx * inv_sx2 + dy * dy * inv_sy2);
      const float w = opacity * __expf(e);
      if (w < 1e-5f) continue;

      const int idx = yy * width + xx;
      atomicAdd(&accum[idx].x, w * cr);
      atomicAdd(&accum[idx].y, w * cg);
      atomicAdd(&accum[idx].z, w * cb);
      atomicAdd(&accum[idx].w, w);
    }
  }
}

__global__ void splat_sliced_kernel(
    const float* means,
    const float* scales,
    const float* colors,
    const float* opacities,
    int n,
    int width,
    int height,
    int slices,
    const float* view,
    const float* proj,
    float4* accum_sliced) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float x = means[i * 3 + 0];
  const float y = means[i * 3 + 1];
  const float z = means[i * 3 + 2];

  const float sx = scales[i * 3 + 0];
  const float sy = scales[i * 3 + 1];

  const float cr = colors[i * 3 + 0];
  const float cg = colors[i * 3 + 1];
  const float cb = colors[i * 3 + 2];
  const float opacity = opacities[i];

  const float4 p_obj = make_float4(x, y, z, 1.0f);
  const float4 p_cam = mat4_mul_vec4_rowmajor(view, p_obj);
  const float z_abs = fabsf(p_cam.z) + 1e-6f;

  const float4 p_clip = mat4_mul_vec4_rowmajor(proj, p_cam);
  if (p_clip.w == 0.0f) return;

  const float inv_w = 1.0f / p_clip.w;
  const float ndc_x = p_clip.x * inv_w;
  const float ndc_y = p_clip.y * inv_w;
  const float ndc_z = p_clip.z * inv_w;
  if (ndc_z < -1.0f || ndc_z > 1.0f) return;

  const float depth01 = ndc_z * 0.5f + 0.5f;  // near=-1 -> 0, far=+1 -> 1
  int slice = static_cast<int>(depth01 * static_cast<float>(slices));
  slice = max(0, min(slices - 1, slice));

  const float px = (ndc_x * 0.5f + 0.5f) * (width - 1);
  const float py = (1.0f - (ndc_y * 0.5f + 0.5f)) * (height - 1);

  const float fx = fabsf(proj[0]);
  const float fy = fabsf(proj[5]);
  float sigma_x = sx * 0.5f * width * fx / z_abs;
  float sigma_y = sy * 0.5f * height * fy / z_abs;
  sigma_x = fmaxf(sigma_x, 1.0f);
  sigma_y = fmaxf(sigma_y, 1.0f);

  const float rad_x = 3.0f * sigma_x;
  const float rad_y = 3.0f * sigma_y;

  const int xmin = max(0, static_cast<int>(floorf(px - rad_x)));
  const int xmax = min(width - 1, static_cast<int>(ceilf(px + rad_x)));
  const int ymin = max(0, static_cast<int>(floorf(py - rad_y)));
  const int ymax = min(height - 1, static_cast<int>(ceilf(py + rad_y)));

  const float inv_sx2 = 1.0f / (sigma_x * sigma_x);
  const float inv_sy2 = 1.0f / (sigma_y * sigma_y);

  const int pixels = width * height;
  for (int yy = ymin; yy <= ymax; ++yy) {
    for (int xx = xmin; xx <= xmax; ++xx) {
      const float dx = (static_cast<float>(xx) + 0.5f) - px;
      const float dy = (static_cast<float>(yy) + 0.5f) - py;
      const float e = -0.5f * (dx * dx * inv_sx2 + dy * dy * inv_sy2);
      float a = opacity * __expf(e);
      if (a < 1e-5f) continue;
      a = clamp01(a);

      const int pidx = yy * width + xx;
      const int idx = slice * pixels + pidx;
      atomicAdd(&accum_sliced[idx].x, a * cr);
      atomicAdd(&accum_sliced[idx].y, a * cg);
      atomicAdd(&accum_sliced[idx].z, a * cb);
      atomicAdd(&accum_sliced[idx].w, a);
    }
  }
}

__global__ void finalize_kernel(
    int width,
    int height,
    float3 background,
    const float4* accum,
    std::uint8_t* out_rgba) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = width * height;
  if (idx >= n) return;

  const float4 a = accum[idx];
  const float denom = 1.0f + a.w;
  float r = (background.x + a.x) / denom;
  float g = (background.y + a.y) / denom;
  float b = (background.z + a.z) / denom;

  r = fminf(fmaxf(r, 0.0f), 1.0f);
  g = fminf(fmaxf(g, 0.0f), 1.0f);
  b = fminf(fmaxf(b, 0.0f), 1.0f);

  const int o = idx * 4;
  out_rgba[o + 0] = static_cast<std::uint8_t>(r * 255.0f + 0.5f);
  out_rgba[o + 1] = static_cast<std::uint8_t>(g * 255.0f + 0.5f);
  out_rgba[o + 2] = static_cast<std::uint8_t>(b * 255.0f + 0.5f);
  out_rgba[o + 3] = 255;
}

__global__ void finalize_sliced_kernel(
    int width,
    int height,
    int slices,
    float3 background,
    const float4* accum_sliced,
    std::uint8_t* out_rgba) {
  const int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixels = width * height;
  if (pidx >= pixels) return;

  float T = 1.0f;
  float3 out = make_float3(0.0f, 0.0f, 0.0f);

  for (int s = 0; s < slices; ++s) {
    const float4 a = accum_sliced[s * pixels + pidx];
    const float sum_alpha = a.w;
    if (sum_alpha > 1e-8f) {
      // a.xyz stores sum(alpha_i * color_i). Convert to average color in [0,1].
      float3 avg;
      avg.x = a.x / sum_alpha;
      avg.y = a.y / sum_alpha;
      avg.z = a.z / sum_alpha;
      avg.x = clamp01(avg.x);
      avg.y = clamp01(avg.y);
      avg.z = clamp01(avg.z);

      // Map summed alpha mass to a bounded effective alpha.
      const float alpha = clamp01(1.0f - __expf(-sum_alpha));
      if (alpha > 0.0f) {
        out.x += T * alpha * avg.x;
        out.y += T * alpha * avg.y;
        out.z += T * alpha * avg.z;
        T *= (1.0f - alpha);
        if (T <= 1e-4f) break;
      }
    }
  }

  out.x += T * background.x;
  out.y += T * background.y;
  out.z += T * background.z;

  out.x = fminf(fmaxf(out.x, 0.0f), 1.0f);
  out.y = fminf(fmaxf(out.y, 0.0f), 1.0f);
  out.z = fminf(fmaxf(out.z, 0.0f), 1.0f);

  const int o = pidx * 4;
  out_rgba[o + 0] = static_cast<std::uint8_t>(out.x * 255.0f + 0.5f);
  out_rgba[o + 1] = static_cast<std::uint8_t>(out.y * 255.0f + 0.5f);
  out_rgba[o + 2] = static_cast<std::uint8_t>(out.z * 255.0f + 0.5f);
  out_rgba[o + 3] = 255;
}

std::vector<std::uint8_t> render_gaussians_cuda(
    const float* means,
    const float* scales,
    const float* colors,
    const float* opacities,
    int n,
    const RenderParams& params) {
  if (n <= 0) {
    return std::vector<std::uint8_t>(params.width * params.height * 4, 0);
  }

  const int width = params.width;
  const int height = params.height;
  const int pixels = width * height;

  struct DeviceBuffers {
    float* means = nullptr;
    float* scales = nullptr;
    float* colors = nullptr;
    float* opacities = nullptr;
    float* view = nullptr;
    float* proj = nullptr;
    float4* accum = nullptr;
    float4* accum_sliced = nullptr;
    std::uint8_t* out = nullptr;
    int cap_n = 0;
    int cap_pixels = 0;
    int cap_slices = 0;

    void ensure(int want_n, int want_pixels, int want_slices) {
      if (want_n > cap_n) {
        if (means) cudaFree(means);
        if (scales) cudaFree(scales);
        if (colors) cudaFree(colors);
        if (opacities) cudaFree(opacities);
        cap_n = want_n;
        GR_CUDA_CHECK(cudaMalloc(&means, static_cast<size_t>(cap_n) * 3 * sizeof(float)));
        GR_CUDA_CHECK(cudaMalloc(&scales, static_cast<size_t>(cap_n) * 3 * sizeof(float)));
        GR_CUDA_CHECK(cudaMalloc(&colors, static_cast<size_t>(cap_n) * 3 * sizeof(float)));
        GR_CUDA_CHECK(cudaMalloc(&opacities, static_cast<size_t>(cap_n) * sizeof(float)));
      }
      if (want_pixels > cap_pixels) {
        if (accum) cudaFree(accum);
        if (accum_sliced) cudaFree(accum_sliced);
        if (out) cudaFree(out);
        cap_pixels = want_pixels;
        GR_CUDA_CHECK(cudaMalloc(&accum, static_cast<size_t>(cap_pixels) * sizeof(float4)));
        GR_CUDA_CHECK(cudaMalloc(&out, static_cast<size_t>(cap_pixels) * 4));
      }

      if (want_slices != cap_slices || (want_slices > 0 && !accum_sliced)) {
        if (accum_sliced) cudaFree(accum_sliced);
        cap_slices = want_slices;
        if (cap_slices > 0 && cap_pixels > 0) {
          GR_CUDA_CHECK(cudaMalloc(&accum_sliced, static_cast<size_t>(cap_pixels) * static_cast<size_t>(cap_slices) * sizeof(float4)));
        } else {
          accum_sliced = nullptr;
        }
      }

      if (!view) GR_CUDA_CHECK(cudaMalloc(&view, sizeof(float) * 16));
      if (!proj) GR_CUDA_CHECK(cudaMalloc(&proj, sizeof(float) * 16));
    }

    ~DeviceBuffers() {
      if (means) cudaFree(means);
      if (scales) cudaFree(scales);
      if (colors) cudaFree(colors);
      if (opacities) cudaFree(opacities);
      if (view) cudaFree(view);
      if (proj) cudaFree(proj);
      if (accum) cudaFree(accum);
      if (accum_sliced) cudaFree(accum_sliced);
      if (out) cudaFree(out);
    }
  };

  static DeviceBuffers buf;

  const size_t means_bytes = static_cast<size_t>(n) * 3 * sizeof(float);
  const size_t scales_bytes = static_cast<size_t>(n) * 3 * sizeof(float);
  const size_t colors_bytes = static_cast<size_t>(n) * 3 * sizeof(float);
  const size_t op_bytes = static_cast<size_t>(n) * sizeof(float);

  const bool depth_sort = (params.enable_depth_sort != 0);
  int slices = depth_sort ? params.depth_slices : 0;
  if (depth_sort && slices <= 0) slices = 16;
  slices = std::max(1, slices);

  buf.ensure(n, pixels, depth_sort ? slices : 0);

  GR_CUDA_CHECK(cudaMemcpy(buf.means, means, means_bytes, cudaMemcpyHostToDevice));
  GR_CUDA_CHECK(cudaMemcpy(buf.scales, scales, scales_bytes, cudaMemcpyHostToDevice));
  GR_CUDA_CHECK(cudaMemcpy(buf.colors, colors, colors_bytes, cudaMemcpyHostToDevice));
  GR_CUDA_CHECK(cudaMemcpy(buf.opacities, opacities, op_bytes, cudaMemcpyHostToDevice));
  GR_CUDA_CHECK(cudaMemcpy(buf.view, params.view, sizeof(float) * 16, cudaMemcpyHostToDevice));
  GR_CUDA_CHECK(cudaMemcpy(buf.proj, params.proj, sizeof(float) * 16, cudaMemcpyHostToDevice));
  if (!depth_sort) {
    GR_CUDA_CHECK(cudaMemset(buf.accum, 0, sizeof(float4) * static_cast<size_t>(pixels)));
  } else {
    GR_CUDA_CHECK(cudaMemset(buf.accum_sliced, 0, sizeof(float4) * static_cast<size_t>(pixels) * static_cast<size_t>(slices)));
  }

  const int threads = 128;
  const int blocks = (n + threads - 1) / threads;
  const float3 bg = make_float3(params.background[0], params.background[1], params.background[2]);
  if (!depth_sort) {
    splat_kernel<<<blocks, threads>>>(
        buf.means, buf.scales, buf.colors, buf.opacities,
        n, width, height,
        buf.view, buf.proj,
        buf.accum);
    GR_CUDA_CHECK(cudaGetLastError());

    const int threads2 = 256;
    const int blocks2 = (pixels + threads2 - 1) / threads2;
    finalize_kernel<<<blocks2, threads2>>>(width, height, bg, buf.accum, buf.out);
    GR_CUDA_CHECK(cudaGetLastError());
  } else {
    splat_sliced_kernel<<<blocks, threads>>>(
        buf.means, buf.scales, buf.colors, buf.opacities,
        n, width, height, slices,
        buf.view, buf.proj,
        buf.accum_sliced);
    GR_CUDA_CHECK(cudaGetLastError());

    const int threads2 = 256;
    const int blocks2 = (pixels + threads2 - 1) / threads2;
    finalize_sliced_kernel<<<blocks2, threads2>>>(width, height, slices, bg, buf.accum_sliced, buf.out);
    GR_CUDA_CHECK(cudaGetLastError());
  }

  std::vector<std::uint8_t> out(static_cast<size_t>(pixels) * 4);
  GR_CUDA_CHECK(cudaMemcpy(out.data(), buf.out, out.size(), cudaMemcpyDeviceToHost));

  return out;
}

}  // namespace gr
