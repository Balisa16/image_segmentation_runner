#pragma once
#include <string>

namespace Segmenter {

enum class ColorOrder { BGR, RGB };

struct Config {
    std::string model_path;
    std::string class_map_path;
    int image_size = 512;
    float alpha = 0.5f;
    int min_pixels = 50;
    ColorOrder model_input_order = ColorOrder::RGB;
    bool normalize_to_unit_range = true;

    [[nodiscard]] bool is_valid() const noexcept {
        return !model_path.empty() && !class_map_path.empty() &&
               image_size > 0 && alpha >= 0.0f && alpha <= 1.0f &&
               min_pixels >= 0;
    }
};
} // namespace Segmenter