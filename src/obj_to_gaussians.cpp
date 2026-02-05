#include "gr/obj_to_gaussians.h"

#include <tiny_obj_loader.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <iostream>

#include "gr/image_io.h"

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
  float ta[2] = {0.0f, 0.0f};
  float tb[2] = {0.0f, 0.0f};
  float tc[2] = {0.0f, 0.0f};
  int mat_id = -1;
  int tex_idx = -1;
  bool has_uv = false;
  float n[3];
  float area = 0.0f;
};

struct ImageRGBA8 {
  int w = 0;
  int h = 0;
  std::vector<std::uint8_t> rgba;
};

static inline float clamp01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

static inline float frac01(float x) {
  const float f = x - std::floor(x);
  return (f < 0.0f) ? (f + 1.0f) : f;
}

static void sample_bilinear_rgb01(const ImageRGBA8& img, float u, float v, float out_rgb[3]) {
  out_rgb[0] = out_rgb[1] = out_rgb[2] = 0.85f;
  if (img.w <= 0 || img.h <= 0 || img.rgba.empty()) return;

  // Wrap UVs; OBJ v=0 is bottom by convention, images use top-left.
  const float uu = frac01(u);
  const float vv = 1.0f - frac01(v);

  const float x = uu * static_cast<float>(img.w - 1);
  const float y = vv * static_cast<float>(img.h - 1);
  const int x0 = std::max(0, std::min(img.w - 1, static_cast<int>(std::floor(x))));
  const int y0 = std::max(0, std::min(img.h - 1, static_cast<int>(std::floor(y))));
  const int x1 = std::min(img.w - 1, x0 + 1);
  const int y1 = std::min(img.h - 1, y0 + 1);
  const float tx = x - static_cast<float>(x0);
  const float ty = y - static_cast<float>(y0);

  auto read = [&](int xx, int yy, int c) -> float {
    const size_t idx = (static_cast<size_t>(yy) * static_cast<size_t>(img.w) + static_cast<size_t>(xx)) * 4u +
                       static_cast<size_t>(c);
    return static_cast<float>(img.rgba[idx]) / 255.0f;
  };

  for (int c = 0; c < 3; ++c) {
    const float p00 = read(x0, y0, c);
    const float p10 = read(x1, y0, c);
    const float p01 = read(x0, y1, c);
    const float p11 = read(x1, y1, c);
    const float a0 = p00 * (1.0f - tx) + p10 * tx;
    const float a1 = p01 * (1.0f - tx) + p11 * tx;
    out_rgb[c] = clamp01(a0 * (1.0f - ty) + a1 * ty);
  }
}

static std::filesystem::path resolve_texture_path(const std::filesystem::path& obj_dir, const std::string& texname) {
  if (texname.empty()) return {};

  std::filesystem::path p(texname);
  if (p.is_absolute() && std::filesystem::exists(p)) return p;

  // Try relative to the OBJ directory.
  std::filesystem::path a = obj_dir / p;
  if (std::filesystem::exists(a)) return a;

  // Common texture folder.
  std::filesystem::path b = obj_dir / "mabo_textures" / p;
  if (std::filesystem::exists(b)) return b;

  // If run from repo root.
  std::filesystem::path c = std::filesystem::path("assets") / "mabo_textures" / p;
  if (std::filesystem::exists(c)) return c;

  return {};
}

GaussiansHost load_obj_as_gaussians(
    const std::string& obj_path,
    float default_scale,
    float default_opacity,
    int num_surface_samples) {
  tinyobj::ObjReaderConfig config;
  {
    // Let tinyobjloader find the .mtl next to the OBJ.
    std::filesystem::path objp(obj_path);
    const std::filesystem::path obj_dir = objp.parent_path();
    config.mtl_search_path = obj_dir.empty() ? std::string() : (obj_dir.string() + std::string("/"));
  }

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
  const auto& materials = reader.GetMaterials();

  // Fallback mapping (material name -> texture file) for assets/mabo_textures.
  // If the MTL already specifies diffuse_texname, that takes precedence.
  const std::unordered_map<std::string, std::string> mat_to_tex = {
      {"mtl_mbdy1062_00", "tex_mbdy1062_00_diff.png"},
      {"mtl_mchr0001_00_cheek", "tex_mchr0001_00_face0_1_cheek0.png"},
      {"mtl_mchr0001_00_eye", "tex_mchr1062_00_eye_diff.png"},
      {"mtl_mchr0001_00_face0", "tex_mchr0001_00_face0_1_diff.png"},
      {"mtl_mchr0001_00_mayu_l", "tex_mchr1062_00_mayu_diff.png"},
      {"mtl_mchr0001_00_mayu_r", "tex_mchr1062_00_mayu_diff.png"},
      {"mtl_mchr0001_00_mouth", "tex_mchr1062_00_mouth_diff.png"},
      {"mtl_mchr1062_00_hair", "tex_mchr1062_00_hair_diff.png"},
      {"mtl_mtail0001_00", "tex_mtail0001_00_1062_diff.png"},
  };

  std::filesystem::path objp(obj_path);
  const std::filesystem::path obj_dir = objp.parent_path();

  std::vector<int> mat_tex_idx;
  mat_tex_idx.resize(materials.size(), -1);
  std::vector<ImageRGBA8> textures;
  std::unordered_map<std::string, int> tex_cache;

  auto load_texture = [&](const std::string& texname) -> int {
    const std::filesystem::path resolved = resolve_texture_path(obj_dir, texname);
    if (resolved.empty()) return -1;

    const std::string key = resolved.string();
    auto it = tex_cache.find(key);
    if (it != tex_cache.end()) return it->second;

    ImageRGBA8 img;
    std::string err;
    if (!load_image_rgba8(key, img.w, img.h, img.rgba, err)) {
      tex_cache[key] = -1;
      return -1;
    }
    const int idx = static_cast<int>(textures.size());
    textures.push_back(std::move(img));
    tex_cache[key] = idx;
    return idx;
  };

  for (size_t mi = 0; mi < materials.size(); ++mi) {
    const auto& m = materials[mi];
    std::string tex = m.diffuse_texname;
    if (tex.empty()) {
      auto it = mat_to_tex.find(m.name);
      if (it != mat_to_tex.end()) tex = it->second;
    }
    mat_tex_idx[mi] = load_texture(tex);
  }

  (void)textures;
  std::vector<Tri> tris;
  tris.reserve(1024);

  auto get_v3 = [&](int vidx, float out[3]) {
    const size_t base = static_cast<size_t>(vidx) * 3u;
    out[0] = attrib.vertices[base + 0];
    out[1] = attrib.vertices[base + 1];
    out[2] = attrib.vertices[base + 2];
  };

  auto get_t2 = [&](int tidx, float out[2]) {
    if (tidx < 0 || attrib.texcoords.empty()) {
      out[0] = 0.0f;
      out[1] = 0.0f;
      return;
    }
    const size_t base = static_cast<size_t>(tidx) * 2u;
    if (base + 1 >= attrib.texcoords.size()) {
      out[0] = 0.0f;
      out[1] = 0.0f;
      return;
    }
    out[0] = attrib.texcoords[base + 0];
    out[1] = attrib.texcoords[base + 1];
  };

  for (const auto& shape : shapes) {
    const auto& mesh = shape.mesh;
    size_t index_offset = 0;

    // Fallback: many OBJ exports name groups like "g mtl_xxx".
    int shape_tex_idx = -1;
    {
      auto it = mat_to_tex.find(shape.name);
      if (it != mat_to_tex.end()) {
        shape_tex_idx = load_texture(it->second);
      }
    }

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

        const int face_mat_id = (f < mesh.material_ids.size()) ? mesh.material_ids[f] : -1;
        const int face_tex_idx =
          (face_mat_id >= 0 && static_cast<size_t>(face_mat_id) < mat_tex_idx.size()) ? mat_tex_idx[static_cast<size_t>(face_mat_id)]
                                                 : shape_tex_idx;

      for (int k = 1; k + 1 < fv; ++k) {
        const tinyobj::index_t i1 = mesh.indices[index_offset + static_cast<size_t>(k)];
        const tinyobj::index_t i2 = mesh.indices[index_offset + static_cast<size_t>(k + 1)];
        if (i1.vertex_index < 0 || i2.vertex_index < 0) continue;

        Tri t;
        t.mat_id = face_mat_id;
        t.tex_idx = face_tex_idx;
        get_v3(i0.vertex_index, t.a);
        get_v3(i1.vertex_index, t.b);
        get_v3(i2.vertex_index, t.c);

        const bool has_uv = (i0.texcoord_index >= 0) && (i1.texcoord_index >= 0) && (i2.texcoord_index >= 0) &&
                            (!attrib.texcoords.empty());
        t.has_uv = has_uv;
        if (has_uv) {
          get_t2(i0.texcoord_index, t.ta);
          get_t2(i1.texcoord_index, t.tb);
          get_t2(i2.texcoord_index, t.tc);
        }

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

  // Diagnostics: how many triangles have UV+texture
  {
    size_t uv_tris = 0;
    size_t tex_tris = 0;
    for (const auto& t : tris) {
      if (t.has_uv) uv_tris++;
      if (t.has_uv && t.tex_idx >= 0) tex_tris++;
    }
    (void)uv_tris;
    (void)tex_tris;
  }

  int tex_color_hits = 0;
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

    // Default color: light gray, optionally sampled from diffuse texture.
    float rgb[3] = {0.85f, 0.85f, 0.85f};
    if (t.has_uv && t.tex_idx >= 0 && static_cast<size_t>(t.tex_idx) < textures.size()) {
        const float tu = t.ta[0] + u * (t.tb[0] - t.ta[0]) + v * (t.tc[0] - t.ta[0]);
        const float tv = t.ta[1] + u * (t.tb[1] - t.ta[1]) + v * (t.tc[1] - t.ta[1]);
        sample_bilinear_rgb01(textures[static_cast<size_t>(t.tex_idx)], tu, tv, rgb);
      tex_color_hits++;
    }

    out.colors[static_cast<size_t>(i) * 3u + 0] = rgb[0];
    out.colors[static_cast<size_t>(i) * 3u + 1] = rgb[1];
    out.colors[static_cast<size_t>(i) * 3u + 2] = rgb[2];

    out.opacities[static_cast<size_t>(i)] = default_opacity;
  }

  (void)tex_color_hits;

  return out;
}

}  // namespace gr
