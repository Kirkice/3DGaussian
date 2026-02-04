// Minimal CUDA translation unit used to force CMake/MSBuild to perform
// the CUDA device link step for executables that link CUDA code only via
// static libraries.

extern "C" void gr_cuda_link_stub() {}
