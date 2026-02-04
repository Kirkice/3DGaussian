#include "gr/obj_to_gaussians.h"

#include <tiny_obj_loader.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace gr {

static inline void cross3(const float a[3], const float b[3], float out[3]) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

static inline float norm3(const float v[3]) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static inline void normalize3(float v[3]) {
  const float n = norm3(v);
  if (n > 1e-12f) {
    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
  }
}

struct Tri {
  float a[3];
  float b[3];
  float c[3];
  float n[3];
  float area = 0.0f;
};

GaussiansHost load_obj_as_gaussians(
    const std::string& obj_path,
    float default_scale,
    float default_opacity,
    int num_surface_samples) {
  tinyobj::ObjReaderConfig config;
  config.mtl_search_path = "";

  tinyobj::ObjReader reader;
  if (!reader.ParseFromFile(obj_path, config)) {
    if (!reader.Error().empty()) {
      throw std::runtime_error("tinyobjloader: " + reader.Error());
    }
    throw std::runtime_error("Failed to load OBJ: " + obj_path);
  }
  if (!reader.Warning().empty()) {
    // Warnings are not fatal; keep going.
  }

  const auto& attrib = reader.GetAttrib();
  if (attrib.vertices.empty()) {
    throw std::runtime_error("OBJ has no vertices: " + obj_path);
  }

  const auto& shapes = reader.GetShapes();
  std::vector<Tri> tris;
  tris.reserve(1024);

  auto get_v3 = [&](int vidx, float out[3]) {
    const size_t base = static_cast<size_t>(vidx) * 3u;
    out[0] = attrib.vertices[base + 0];
    out[1] = attrib.vertices[base + 1];
    out[2] = attrib.vertices[base + 2];
  };

  for (const auto& shape : shapes) {
    const auto& mesh = shape.mesh;
    size_t index_offset = 0;
    for (size_t f = 0; f < mesh.num_face_vertices.size(); ++f) {
      const int fv = mesh.num_face_vertices[f];
      if (fv < 3) {
        index_offset += static_cast<size_t>(fv);
        continue;
      }

      const tinyobj::index_t i0 = mesh.indices[index_offset + 0];
      if (i0.vertex_index < 0) {
        index_offset += static_cast<size_t>(fv);
        continue;
      }

      for (int k = 1; k + 1 < fv; ++k) {
        const tinyobj::index_t i1 = mesh.indices[index_offset + static_cast<size_t>(k)];
        const tinyobj::index_t i2 = mesh.indices[index_offset + static_cast<size_t>(k + 1)];
        if (i1.vertex_index < 0 || i2.vertex_index < 0) continue;

        Tri t;
        get_v3(i0.vertex_index, t.a);
        get_v3(i1.vertex_index, t.b);
        get_v3(i2.vertex_index, t.c);

        float ab[3] = {t.b[0] - t.a[0], t.b[1] - t.a[1], t.b[2] - t.a[2]};
        float ac[3] = {t.c[0] - t.a[0], t.c[1] - t.a[1], t.c[2] - t.a[2]};
        cross3(ab, ac, t.n);
        const float twice_area = norm3(t.n);
        t.area = 0.5f * twice_area;
        if (t.area <= 1e-12f) continue;
        normalize3(t.n);

        tris.push_back(t);
      }

      index_offset += static_cast<size_t>(fv);
    }
  }

  // Fallback: if there are no faces, just do 1 Gaussian per vertex.
  if (tris.empty() || num_surface_samples <= 0) {
    const size_t num_vertices = attrib.vertices.size() / 3;

    GaussiansHost out;
    out.means.resize(num_vertices * 3);
    out.scales.resize(num_vertices * 3);
    out.colors.resize(num_vertices * 3);
    out.opacities.resize(num_vertices);

    for (size_t i = 0; i < num_vertices; ++i) {
      out.means[i * 3 + 0] = attrib.vertices[i * 3 + 0];
      out.means[i * 3 + 1] = attrib.vertices[i * 3 + 1];
      out.means[i * 3 + 2] = attrib.vertices[i * 3 + 2];

      out.scales[i * 3 + 0] = default_scale;
      out.scales[i * 3 + 1] = default_scale;
      out.scales[i * 3 + 2] = default_scale;

      out.colors[i * 3 + 0] = 0.85f;
      out.colors[i * 3 + 1] = 0.85f;
      out.colors[i * 3 + 2] = 0.85f;

      out.opacities[i] = default_opacity;
    }

    return out;
  }

  // Build area CDF.
  std::vector<float> cdf;
  cdf.resize(tris.size());
  float total_area = 0.0f;
  for (size_t i = 0; i < tris.size(); ++i) {
    total_area += tris[i].area;
    cdf[i] = total_area;
  }
  if (total_area <= 0.0f) {
    throw std::runtime_error("OBJ has degenerate faces (zero area): " + obj_path);
  }

  // Deterministic RNG per file path.
  const std::uint64_t seed64 = static_cast<std::uint64_t>(std::hash<std::string>{}(obj_path));
  std::mt19937 rng(static_cast<std::uint32_t>(seed64 ^ (seed64 >> 32)));
  std::uniform_real_distribution<float> uni01(0.0f, 1.0f);

  const int n = num_surface_samples;
  GaussiansHost out;
  out.means.resize(static_cast<size_t>(n) * 3u);
  out.scales.resize(static_cast<size_t>(n) * 3u);
  out.colors.resize(static_cast<size_t>(n) * 3u);
  out.opacities.resize(static_cast<size_t>(n));

  for (int i = 0; i < n; ++i) {
    const float r = uni01(rng) * total_area;
    const auto it = std::lower_bound(cdf.begin(), cdf.end(), r);
    const size_t tidx = static_cast<size_t>(std::distance(cdf.begin(), it));
    const Tri& t = tris[std::min(tidx, tris.size() - 1)];

    // Uniform sample on triangle via barycentric coordinates.
    float u = uni01(rng);
    float v = uni01(rng);
    if (u + v > 1.0f) {
      u = 1.0f - u;
      v = 1.0f - v;
    }

    const float p[3] = {
        t.a[0] + u * (t.b[0] - t.a[0]) + v * (t.c[0] - t.a[0]),
        t.a[1] + u * (t.b[1] - t.a[1]) + v * (t.c[1] - t.a[1]),
        t.a[2] + u * (t.b[2] - t.a[2]) + v * (t.c[2] - t.a[2]),
    };

    out.means[static_cast<size_t>(i) * 3u + 0] = p[0];
    out.means[static_cast<size_t>(i) * 3u + 1] = p[1];
    out.means[static_cast<size_t>(i) * 3u + 2] = p[2];

    out.scales[static_cast<size_t>(i) * 3u + 0] = default_scale;
    out.scales[static_cast<size_t>(i) * 3u + 1] = default_scale;
    out.scales[static_cast<size_t>(i) * 3u + 2] = default_scale;

    // Default color: light gray.
    out.colors[static_cast<size_t>(i) * 3u + 0] = 0.85f;
    out.colors[static_cast<size_t>(i) * 3u + 1] = 0.85f;
    out.colors[static_cast<size_t>(i) * 3u + 2] = 0.85f;

    out.opacities[static_cast<size_t>(i)] = default_opacity;
  }

  return out;
}

}  // namespace gr
