#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <raylib.h>

#include <cnpy.h>

#include "gr/gaussian_types.h"
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

static int parse_int_arg(int argc, char** argv, const char* key, int def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == key) return std::atoi(argv[i + 1]);
  }
  return def;
}

static float parse_float_arg(int argc, char** argv, const char* key, float def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == key) return static_cast<float>(std::atof(argv[i + 1]));
  }
  return def;
}

static bool load_gaussians_npz(const std::string& path, gr::GaussiansHost& g, std::string& err) {
  try {
    cnpy::npz_t npz = cnpy::npz_load(path);
    if (npz.count("means") == 0 || npz.count("scales") == 0 || npz.count("colors") == 0 || npz.count("opacities") == 0) {
      err = "npz missing required arrays: means/scales/colors/opacities";
      return false;
    }

    cnpy::NpyArray a_means = npz.at("means");
    cnpy::NpyArray a_scales = npz.at("scales");
    cnpy::NpyArray a_colors = npz.at("colors");
    cnpy::NpyArray a_opacities = npz.at("opacities");

    if (a_means.word_size != 4 || a_scales.word_size != 4 || a_colors.word_size != 4 || a_opacities.word_size != 4) {
      err = "npz arrays must be float32";
      return false;
    }
    if (a_means.shape.size() != 2 || a_means.shape[1] != 3) {
      err = "means must be shape (N,3)";
      return false;
    }
    if (a_scales.shape.size() != 2 || a_scales.shape[1] != 3) {
      err = "scales must be shape (N,3)";
      return false;
    }
    if (a_colors.shape.size() != 2 || a_colors.shape[1] != 3) {
      err = "colors must be shape (N,3)";
      return false;
    }

    const size_t n = a_means.shape[0];
    size_t n_op = 0;
    if (a_opacities.shape.size() == 1) {
      n_op = a_opacities.shape[0];
    } else if (a_opacities.shape.size() == 2 && a_opacities.shape[1] == 1) {
      n_op = a_opacities.shape[0];
    } else {
      err = "opacities must be shape (N,) or (N,1)";
      return false;
    }

    if (a_scales.shape[0] != n || a_colors.shape[0] != n || n_op != n) {
      err = "means/scales/colors/opacities N mismatch";
      return false;
    }

    const float* p_means = a_means.data<float>();
    const float* p_scales = a_scales.data<float>();
    const float* p_colors = a_colors.data<float>();
    const float* p_opacities = a_opacities.data<float>();

    g.means.assign(p_means, p_means + n * 3);
    g.scales.assign(p_scales, p_scales + n * 3);
    g.colors.assign(p_colors, p_colors + n * 3);
    g.opacities.assign(p_opacities, p_opacities + n);
    return true;
  } catch (const std::exception& e) {
    err = e.what();
    return false;
  }
}

int main(int argc, char** argv) {
  const std::string npz = (argc >= 2) ? std::string(argv[1]) : std::string("outputs/fit_scene_tex_m1/gaussians_fitted.npz");
  const int width = parse_int_arg(argc, argv, "--width", 960);
  const int height = parse_int_arg(argc, argv, "--height", 540);
  const float fovy = parse_float_arg(argc, argv, "--fovy", 60.0f);
  const int max_gaussians = parse_int_arg(argc, argv, "--max", 1000000);

  gr::GaussiansHost g;
  std::string err;
  if (!load_gaussians_npz(npz, g, err)) {
    std::cerr << "Failed to load npz: " << npz << " (" << err << ")\n";
    return 1;
  }

  int n = g.count();
  if (max_gaussians > 0) n = std::min(n, max_gaussians);
  if (n <= 0) {
    std::cerr << "No gaussians loaded.\n";
    return 1;
  }

  std::cout << "Loaded gaussians: " << n << " from " << npz << "\n";
  std::cout << "Controls: LMB-drag orbit, wheel zoom, R reset, H toggle HUD, ESC quit\n";

  SetConfigFlags(FLAG_VSYNC_HINT);
  InitWindow(width, height, "Gaussian Native Viewer");

  Image img;
  img.data = nullptr;
  img.width = width;
  img.height = height;
  img.mipmaps = 1;
  img.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
  Texture2D tex = LoadTextureFromImage(img);

  float yaw = 0.0f;
  float pitch = 0.2f;
  float radius = 2.5f;
  bool show_hud = true;

  gr::RenderParams params;
  params.width = width;
  params.height = height;
  params.background[0] = 0.02f;
  params.background[1] = 0.02f;
  params.background[2] = 0.02f;
  params.enable_depth_sort = 1;
  params.depth_slices = 32;

  const float target[3] = {0.0f, 0.0f, 0.0f};
  const float up[3] = {0.0f, 1.0f, 0.0f};

  double last_time = GetTime();
  int frame_count = 0;
  float fps_smooth = 0.0f;

  while (!WindowShouldClose()) {
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
      Vector2 delta = GetMouseDelta();
      yaw += delta.x * 0.01f;
      pitch += delta.y * 0.01f;
      pitch = std::max(-1.4f, std::min(1.4f, pitch));
    }
    float wheel = GetMouseWheelMove();
    if (wheel != 0.0f) {
      radius *= std::pow(0.9f, wheel);
      radius = std::max(0.2f, std::min(50.0f, radius));
    }
    if (IsKeyPressed(KEY_R)) {
      yaw = 0.0f;
      pitch = 0.2f;
      radius = 2.5f;
    }
    if (IsKeyPressed(KEY_H)) {
      show_hud = !show_hud;
    }

    float eye[3];
    eye[0] = radius * std::cos(pitch) * std::sin(yaw);
    eye[1] = radius * std::sin(pitch);
    eye[2] = radius * std::cos(pitch) * std::cos(yaw);

    look_at(eye, target, up, params.view);
    perspective(fovy, static_cast<float>(width) / static_cast<float>(height), 0.01f, 100.0f, params.proj);

    std::vector<std::uint8_t> rgba = gr::render_gaussians(
        g.means.data(), g.scales.data(), g.colors.data(), g.opacities.data(), n, params);

    UpdateTexture(tex, rgba.data());

    frame_count++;
    const double now = GetTime();
    const double dt = now - last_time;
    if (dt >= 0.25) {
      const float fps = static_cast<float>(frame_count / dt);
      fps_smooth = (fps_smooth == 0.0f) ? fps : (0.8f * fps_smooth + 0.2f * fps);
      frame_count = 0;
      last_time = now;
    }

    BeginDrawing();
    ClearBackground(BLACK);
    DrawTexture(tex, 0, 0, WHITE);
    if (show_hud) {
      DrawRectangle(10, 10, 620, 70, Fade(BLACK, 0.6f));
      const char* backend = (params.force_cpu != 0) ? "CPU" : ((GR_CUDA_ENABLED != 0) ? "CUDA" : "CPU");
      DrawText(TextFormat("Backend: %s  Gaussians: %d  FPS: %.1f", backend, n, fps_smooth), 20, 20, 20, RAYWHITE);
      DrawText("LMB orbit | Wheel zoom | R reset | H HUD | ESC quit", 20, 45, 18, RAYWHITE);
    }
    EndDrawing();
  }

  UnloadTexture(tex);
  CloseWindow();
  return 0;
}
