#pragma once

#include <string>

#include "gr/gaussian_types.h"

namespace gr {

GaussiansHost load_obj_as_gaussians(
    const std::string& obj_path,
    float default_scale,
    float default_opacity,
    int num_surface_samples = 200000);

}  // namespace gr
