#include "gr/colors_io.h"

#include <cstdint>
#include <fstream>
#include <limits>

namespace gr {

static bool read_exact(std::ifstream& f, void* dst, size_t bytes) {
  f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
  return f.good();
}

bool load_colors_bin(const std::string& path, int expected_n, std::vector<float>& colors_out, std::string& error_out) {
  error_out.clear();
  colors_out.clear();

  if (expected_n <= 0) {
    error_out = "expected_n must be > 0";
    return false;
  }

  std::ifstream f(path, std::ios::binary);
  if (!f) {
    error_out = "failed to open file";
    return false;
  }

  char magic[4] = {0, 0, 0, 0};
  if (!read_exact(f, magic, sizeof(magic))) {
    error_out = "failed to read magic";
    return false;
  }
  if (!(magic[0] == 'G' && magic[1] == 'R' && magic[2] == 'C' && magic[3] == 'L')) {
    error_out = "bad magic (expected 'GRCL')";
    return false;
  }

  std::uint32_t n_u32 = 0;
  if (!read_exact(f, &n_u32, sizeof(n_u32))) {
    error_out = "failed to read N";
    return false;
  }

  if (n_u32 == 0 || n_u32 > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    error_out = "invalid N in file";
    return false;
  }
  const int n = static_cast<int>(n_u32);
  if (n != expected_n) {
    error_out = "N mismatch (file N != expected N)";
    return false;
  }

  const size_t count = static_cast<size_t>(n) * 3u;
  colors_out.resize(count);
  const size_t bytes = count * sizeof(float);
  if (!read_exact(f, colors_out.data(), bytes)) {
    error_out = "failed to read color payload";
    colors_out.clear();
    return false;
  }

  return true;
}

}  // namespace gr
