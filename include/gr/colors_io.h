#pragma once

#include <string>
#include <vector>

namespace gr {

// Simple binary format for per-Gaussian colors.
// Layout (little-endian):
//   4 bytes: magic 'G''R''C''L'
//   uint32:  N
//   float32: colors[N*3] (RGB, linear 0..1)
//
// Returns true on success; on failure returns false and sets error_out.
bool load_colors_bin(const std::string& path, int expected_n, std::vector<float>& colors_out, std::string& error_out);

}  // namespace gr
