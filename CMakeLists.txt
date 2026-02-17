cmake_minimum_required(VERSION 3.24)

project(GaussianRenderer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(GR_ENABLE_CUDA "Enable CUDA renderer" ON)
option(GR_BUILD_PYTHON "Build Python extension module" ON)
option(GR_BUILD_NATIVE_VIEWER "Build realtime Gaussian model viewer" ON)

if(GR_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development QUIET)
  if(NOT Python_FOUND)
    message(STATUS "Python not found (Interpreter+Development). Building without Python module; set -DPython_ROOT_DIR=... to enable.")
    set(GR_BUILD_PYTHON OFF)
  endif()
endif()

include(FetchContent)

if(GR_BUILD_PYTHON)
  # Use CMake's modern FindPython.
  set(PYBIND11_FINDPYTHON ON)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
  )
  FetchContent_MakeAvailable(pybind11)
endif()

if(GR_BUILD_NATIVE_VIEWER)
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

  set(GR_RAYLIB_TARGET raylib)
  if(EXISTS "${CMAKE_SOURCE_DIR}/vendor/raylib/CMakeLists.txt")
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
    add_subdirectory(vendor/raylib EXCLUDE_FROM_ALL)

    if(TARGET raylib::raylib)
      set(GR_RAYLIB_TARGET raylib::raylib)
    elseif(TARGET raylib)
      set(GR_RAYLIB_TARGET raylib)
    elseif(TARGET raylib_static)
      set(GR_RAYLIB_TARGET raylib_static)
    endif()
  else()
    find_package(raylib QUIET)

    if(TARGET raylib::raylib)
      set(GR_RAYLIB_TARGET raylib::raylib)
    elseif(TARGET raylib)
      set(GR_RAYLIB_TARGET raylib)
    elseif(TARGET raylib_static)
      set(GR_RAYLIB_TARGET raylib_static)
    else()
      set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
      FetchContent_Declare(
        raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG ae50bfa2cc569c0f8d5bc4315d39db64005b1b08
      )
      set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
      set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
      FetchContent_MakeAvailable(raylib)
    endif()
  endif()

  find_package(ZLIB REQUIRED)
  FetchContent_Declare(
    cnpy
    GIT_REPOSITORY https://github.com/rogersce/cnpy.git
    GIT_TAG master
  )
  FetchContent_MakeAvailable(cnpy)
endif()

set(GR_CORE_SOURCES
  src/renderer_cpu.cpp
  src/renderer_dispatch.cpp
)

if(GR_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # Prefer static CUDA runtime to avoid needing CUDA\bin on PATH at runtime.
  # (On Windows, missing cudart64_12.dll typically shows as exit code -1073741515.)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

  # With Visual Studio generators, MSBuild's CUDA targets sometimes fail to
  # auto-detect the CUDA toolkit location. If CUDA_PATH is set, pass it through
  # explicitly as a global property.
  if(MSVC AND CMAKE_GENERATOR MATCHES "Visual Studio")
    unset(_gr_cuda_path)
    if(DEFINED CUDAToolkit_ROOT)
      set(_gr_cuda_path "${CUDAToolkit_ROOT}")
    elseif(DEFINED ENV{CUDA_PATH})
      set(_gr_cuda_path "$ENV{CUDA_PATH}")
    elseif(DEFINED CMAKE_CUDA_COMPILER)
      # e.g. .../CUDA/v12.6/bin/nvcc.exe -> .../CUDA/v12.6
      get_filename_component(_gr_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
      get_filename_component(_gr_cuda_path "${_gr_cuda_bin}" DIRECTORY)
    endif()

    if(DEFINED _gr_cuda_path)
      file(TO_CMAKE_PATH "${_gr_cuda_path}" _gr_cuda_path)
      list(APPEND CMAKE_VS_GLOBALS "CudaToolkitDir=${_gr_cuda_path}\\")
    endif()
  endif()

  # If the user didn't specify an architecture, try to build for the local GPU.
  # You can override this with -DCMAKE_CUDA_ARCHITECTURES=86 (or similar).
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA archs" FORCE)
  endif()

  find_package(CUDAToolkit REQUIRED)
  list(APPEND GR_CORE_SOURCES src/renderer.cu)
else()
  message(STATUS "GR_ENABLE_CUDA=OFF: building CPU renderer")
endif()

add_library(gaussian_core STATIC ${GR_CORE_SOURCES})
target_include_directories(gaussian_core PUBLIC include)

if(GR_ENABLE_CUDA)
  target_compile_definitions(gaussian_core PUBLIC GR_CUDA_ENABLED=1)
else()
  target_compile_definitions(gaussian_core PUBLIC GR_CUDA_ENABLED=0)
endif()

if(GR_ENABLE_CUDA)
  if(TARGET CUDA::cudart_static)
    target_link_libraries(gaussian_core PUBLIC CUDA::cudart_static)
  else()
    target_link_libraries(gaussian_core PUBLIC CUDA::cudart)
  endif()
  set_target_properties(gaussian_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(GR_BUILD_PYTHON)
  pybind11_add_module(gaussian_renderer MODULE src/bindings.cpp)
  target_link_libraries(gaussian_renderer PRIVATE gaussian_core)
  target_compile_definitions(gaussian_renderer PRIVATE _USE_MATH_DEFINES)
  if(GR_ENABLE_CUDA)
    target_sources(gaussian_renderer PRIVATE src/cuda_link_stub.cu)
    set_target_properties(gaussian_renderer PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
  endif()
endif()

if(GR_BUILD_NATIVE_VIEWER)
  add_executable(gaussian_native_viewer src/model_viewer_main.cpp)
  target_link_libraries(gaussian_native_viewer PRIVATE gaussian_core ${GR_RAYLIB_TARGET} cnpy ZLIB::ZLIB)
  target_include_directories(gaussian_native_viewer PRIVATE ${cnpy_SOURCE_DIR})
  if(GR_ENABLE_CUDA)
    target_sources(gaussian_native_viewer PRIVATE src/cuda_link_stub.cu)
    set_target_properties(gaussian_native_viewer PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
  endif()
endif()

# Make MSVC + NVCC a bit happier with modern C++.
if(MSVC)
  # IMPORTANT: Don't pass MSVC-only flags (like /EHsc) to CUDA compilation.
  # With Visual Studio generators, CUDA build customization can forward these
  # directly to nvcc, which then errors out.
  target_compile_options(gaussian_core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
  if(TARGET gaussian_renderer)
    target_compile_options(gaussian_renderer PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
  endif()
  if(TARGET gaussian_native_viewer)
    target_compile_options(gaussian_native_viewer PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
  endif()
endif()
