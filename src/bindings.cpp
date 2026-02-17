#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "gr/gaussian_types.h"
#include "gr/renderer.h"

namespace py = pybind11;

static void require_contiguous_f32(const py::array& arr, const char* name) {
  if (!py::isinstance<py::array>(arr)) {
    throw std::runtime_error(std::string(name) + " must be a numpy array");
  }
  if (arr.dtype().kind() != 'f' || arr.itemsize() != 4) {
    throw std::runtime_error(std::string(name) + " must be float32");
  }
  if (!(arr.flags() & py::array::c_style)) {
    throw std::runtime_error(std::string(name) + " must be C-contiguous");
  }
}

PYBIND11_MODULE(gaussian_renderer, m) {
  m.doc() = "3D Gaussian renderer core bindings";

  m.def(
      "render_gaussians",
      [](py::array means,
         py::array scales,
         py::array colors,
         py::array opacities,
         int width,
         int height,
         py::array view,
         py::array proj,
         py::object background) {
        require_contiguous_f32(means, "means");
        require_contiguous_f32(scales, "scales");
        require_contiguous_f32(colors, "colors");
        require_contiguous_f32(opacities, "opacities");
        require_contiguous_f32(view, "view");
        require_contiguous_f32(proj, "proj");

        if (means.ndim() != 2 || means.shape(1) != 3) throw std::runtime_error("means must be (N,3)");
        if (scales.ndim() != 2 || scales.shape(1) != 3) throw std::runtime_error("scales must be (N,3)");
        if (colors.ndim() != 2 || colors.shape(1) != 3) throw std::runtime_error("colors must be (N,3)");
        if (opacities.ndim() != 1) throw std::runtime_error("opacities must be (N,)");
        if (view.ndim() != 2 || view.shape(0) != 4 || view.shape(1) != 4) throw std::runtime_error("view must be (4,4)");
        if (proj.ndim() != 2 || proj.shape(0) != 4 || proj.shape(1) != 4) throw std::runtime_error("proj must be (4,4)");
        py::array bg_arr;
        if (background.is_none()) {
          bg_arr = py::array_t<float>({3});
          auto* p = static_cast<float*>(bg_arr.mutable_data());
          p[0] = 0.0f;
          p[1] = 0.0f;
          p[2] = 0.0f;
        } else {
          bg_arr = py::cast<py::array>(background);
          require_contiguous_f32(bg_arr, "background");
          if (bg_arr.ndim() != 1 || bg_arr.shape(0) != 3) throw std::runtime_error("background must be (3,)");
        }

        const int n = static_cast<int>(means.shape(0));
        if (scales.shape(0) != n || colors.shape(0) != n || opacities.shape(0) != n) {
          throw std::runtime_error("means/scales/colors/opacities must have matching N");
        }

        gr::RenderParams params;
        params.width = width;
        params.height = height;
        std::memcpy(params.view, view.data(), sizeof(float) * 16);
        std::memcpy(params.proj, proj.data(), sizeof(float) * 16);
        std::memcpy(params.background, bg_arr.data(), sizeof(float) * 3);

        std::vector<std::uint8_t> rgba = gr::render_gaussians(
            static_cast<const float*>(means.data()),
            static_cast<const float*>(scales.data()),
            static_cast<const float*>(colors.data()),
            static_cast<const float*>(opacities.data()),
            n,
            params);

        // Return as (H,W,4) uint8
        py::array_t<std::uint8_t> out({height, width, 4});
        std::memcpy(out.mutable_data(), rgba.data(), rgba.size());
        return out;
      },
      py::arg("means"),
      py::arg("scales"),
      py::arg("colors"),
      py::arg("opacities"),
      py::arg("width") = 800,
      py::arg("height") = 600,
      py::arg("view"),
      py::arg("proj"),
      py::arg("background") = py::none());
}
