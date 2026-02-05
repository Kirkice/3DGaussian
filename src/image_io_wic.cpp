#include "gr/image_io.h"

#include <algorithm>

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>

#include <wincodec.h>

#include <mutex>

namespace gr {

static std::once_flag g_wic_once;
static IWICImagingFactory* g_wic_factory = nullptr;
static std::string g_wic_error;

static void init_wic() {
  HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
    g_wic_error = "CoInitializeEx failed";
    return;
  }

  IWICImagingFactory* factory = nullptr;
  hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
  if (FAILED(hr) || factory == nullptr) {
    g_wic_error = "CoCreateInstance(CLSID_WICImagingFactory) failed";
    return;
  }

  g_wic_factory = factory;
}

static std::wstring to_wstring(const std::string& s) {
  if (s.empty()) return std::wstring();
  const int needed = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  if (needed <= 0) {
    // Fallback: treat input as ANSI.
    const int needed_ansi = MultiByteToWideChar(CP_ACP, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w;
    w.resize(static_cast<size_t>(std::max(0, needed_ansi - 1)));
    if (needed_ansi > 0) MultiByteToWideChar(CP_ACP, 0, s.c_str(), -1, w.data(), needed_ansi);
    return w;
  }
  std::wstring w;
  w.resize(static_cast<size_t>(needed - 1));
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), needed);
  return w;
}

bool load_image_rgba8(const std::string& path, int& width_out, int& height_out, std::vector<std::uint8_t>& rgba_out,
                      std::string& error_out) {
  width_out = 0;
  height_out = 0;
  rgba_out.clear();
  error_out.clear();

  std::call_once(g_wic_once, init_wic);
  if (g_wic_factory == nullptr) {
    error_out = !g_wic_error.empty() ? g_wic_error : "WIC factory not initialized";
    return false;
  }

  const std::wstring wpath = to_wstring(path);
  if (wpath.empty()) {
    error_out = "empty path";
    return false;
  }

  IWICBitmapDecoder* decoder = nullptr;
  HRESULT hr = g_wic_factory->CreateDecoderFromFilename(wpath.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad,
                                                        &decoder);
  if (FAILED(hr) || decoder == nullptr) {
    error_out = "CreateDecoderFromFilename failed";
    return false;
  }

  IWICBitmapFrameDecode* frame = nullptr;
  hr = decoder->GetFrame(0, &frame);
  if (FAILED(hr) || frame == nullptr) {
    decoder->Release();
    error_out = "GetFrame(0) failed";
    return false;
  }

  UINT w = 0, h = 0;
  hr = frame->GetSize(&w, &h);
  if (FAILED(hr) || w == 0 || h == 0) {
    frame->Release();
    decoder->Release();
    error_out = "GetSize failed";
    return false;
  }

  IWICFormatConverter* conv = nullptr;
  hr = g_wic_factory->CreateFormatConverter(&conv);
  if (FAILED(hr) || conv == nullptr) {
    frame->Release();
    decoder->Release();
    error_out = "CreateFormatConverter failed";
    return false;
  }

  hr = conv->Initialize(frame, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0.0,
                        WICBitmapPaletteTypeCustom);
  if (FAILED(hr)) {
    conv->Release();
    frame->Release();
    decoder->Release();
    error_out = "FormatConverter::Initialize failed";
    return false;
  }

  const size_t stride = static_cast<size_t>(w) * 4u;
  const size_t bytes = stride * static_cast<size_t>(h);
  rgba_out.resize(bytes);

  hr = conv->CopyPixels(nullptr, static_cast<UINT>(stride), static_cast<UINT>(bytes), rgba_out.data());
  conv->Release();
  frame->Release();
  decoder->Release();

  if (FAILED(hr)) {
    rgba_out.clear();
    error_out = "CopyPixels failed";
    return false;
  }

  width_out = static_cast<int>(w);
  height_out = static_cast<int>(h);
  return true;
}

}  // namespace gr

#else

namespace gr {
bool load_image_rgba8(const std::string&, int& width_out, int& height_out, std::vector<std::uint8_t>& rgba_out,
                      std::string& error_out) {
  width_out = 0;
  height_out = 0;
  rgba_out.clear();
  error_out = "load_image_rgba8 is only implemented on Windows (WIC)";
  return false;
}
}  // namespace gr

#endif
