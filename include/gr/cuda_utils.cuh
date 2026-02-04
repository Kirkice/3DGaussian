#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace gr {

inline void cuda_check(cudaError_t err, const char* file, int line) {
  if (err == cudaSuccess) return;
  std::ostringstream oss;
  oss << "CUDA error: " << cudaGetErrorString(err) << " (" << static_cast<int>(err)
      << ") at " << file << ":" << line;
  throw std::runtime_error(oss.str());
}

#define GR_CUDA_CHECK(x) ::gr::cuda_check((x), __FILE__, __LINE__)

}  // namespace gr
