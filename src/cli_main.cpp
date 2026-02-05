#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gr/colors_io.h"
#include "gr/obj_to_gaussians.h"
#include "gr/renderer.h"

static void perspective(float fovy_deg, float aspect, float znear, float zfar, float out[16]) {
  const float f = 1.0f / std::tan(fovy_deg * 3.1415926535f / 180.0f * 0.5f);
  for (int i = 0; i < 16; ++i) out[i] = 0.0f;
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (zfar + znear) / (znear - zfar);
  out[11] = (2.0f * zfar * znear) / (znear - zfar);
  out[14] = -1.0f;
}

static void normalize3(float v[3]) {
  const float n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + 1e-8f;
  v[0] /= n;
  v[1] /= n;
  v[2] /= n;
}

static void cross3(const float a[3], const float b[3], float out[3]) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

static float dot3(const float a[3], const float b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void look_at(const float eye[3], const float target[3], const float up[3], float out[16]) {
  float f[3] = {target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]};
  normalize3(f);

  float u[3] = {up[0], up[1], up[2]};
  normalize3(u);

  float s[3];
  cross3(f, u, s);
  normalize3(s);

  float u2[3];
  cross3(s, f, u2);

  // Row-major
  out[0] = s[0];
  out[1] = s[1];
  out[2] = s[2];
  out[3] = -dot3(s, eye);

  out[4] = u2[0];
  out[5] = u2[1];
  out[6] = u2[2];
  out[7] = -dot3(u2, eye);

  out[8] = -f[0];
  out[9] = -f[1];
  out[10] = -f[2];
  out[11] = dot3(f, eye);

  out[12] = 0.0f;
  out[13] = 0.0f;
  out[14] = 0.0f;
  out[15] = 1.0f;
}

static bool write_ppm(const std::string& path, int w, int h, const std::vector<std::uint8_t>& rgba) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f << "P6\n" << w << " " << h << "\n255\n";
  for (int i = 0; i < w * h; ++i) {
    f.put(static_cast<char>(rgba[i * 4 + 0]));
    f.put(static_cast<char>(rgba[i * 4 + 1]));
    f.put(static_cast<char>(rgba[i * 4 + 2]));
  }
  return true;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: gaussian_render_cli <model.obj> <out.ppm> [width height [samples]] [--colors_bin path] \n";
    return 2;
  }
  const std::string obj = argv[1];
  const std::string out = argv[2];
  const int width = (argc >= 4) ? std::atoi(argv[3]) : 960;
  const int height = (argc >= 5) ? std::atoi(argv[4]) : 540;
  const int samples = (argc >= 6) ? std::atoi(argv[5]) : 200000;

  auto parse_colors_bin = [&](int argc2, char** argv2) -> std::string {
    for (int i = 1; i + 1 < argc2; ++i) {
      if (std::string(argv2[i]) == "--colors_bin") return std::string(argv2[i + 1]);
    }
    return std::string();
  };
  const std::string colors_bin = parse_colors_bin(argc, argv);

  auto parse_int_flag = [&](const char* key, int def) -> int {
    for (int i = 1; i + 1 < argc; ++i) {
      if (std::string(argv[i]) == key) return std::atoi(argv[i + 1]);
    }
    return def;
  };
  const int enable_sort = parse_int_flag("--sort", 0);
  const int sort_slices = parse_int_flag("--slices", 16);
  const int true_sort = parse_int_flag("--true_sort", 0);

  try {
    gr::GaussiansHost g = gr::load_obj_as_gaussians(obj, 0.01f, 0.8f, samples);

    if (!colors_bin.empty()) {
      std::string err;
      std::vector<float> colors;
      if (!gr::load_colors_bin(colors_bin, g.count(), colors, err)) {
        std::cerr << "Failed to load colors_bin: " << colors_bin << " (" << err << ")\n";
        return 1;
      }
      g.colors = std::move(colors);
      std::cout << "Loaded colors override: " << colors_bin << "\n";
    }

    gr::RenderParams params;
    params.width = width;
    params.height = height;
    params.background[0] = 0.02f;
    params.background[1] = 0.02f;
    params.background[2] = 0.02f;
    params.enable_depth_sort = (true_sort != 0) ? 1 : enable_sort;
    params.depth_slices = sort_slices;
    params.force_cpu = (true_sort != 0) ? 1 : 0;

    const float eye[3] = {0.0f, 0.0f, 2.5f};
    const float target[3] = {0.0f, 0.0f, 0.0f};
    const float up[3] = {0.0f, 1.0f, 0.0f};
    look_at(eye, target, up, params.view);
    perspective(60.0f, static_cast<float>(width) / static_cast<float>(height), 0.01f, 100.0f, params.proj);

    std::vector<std::uint8_t> rgba = gr::render_gaussians(
        g.means.data(), g.scales.data(), g.colors.data(), g.opacities.data(), g.count(), params);

    if (!write_ppm(out, width, height, rgba)) {
      std::cerr << "Failed to write " << out << "\n";
      return 1;
    }

    std::cout << "Wrote " << out << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
