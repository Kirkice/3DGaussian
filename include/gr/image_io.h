#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gr {

// Loads an image file (PNG recommended) into RGBA8.
// Returns true on success; on failure returns false and sets error_out.
bool load_image_rgba8(const std::string& path, int& width_out, int& height_out, std::vector<std::uint8_t>& rgba_out,
                      std::string& error_out);

}  // namespace gr
