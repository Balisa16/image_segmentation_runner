#pragma once

#include "segmenter/config.hpp"
#include "segmenter/result.hpp"

#include <chrono>
#include <map>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>

namespace Segmenter {

enum class ComputeDevice { CPU, CUDA, CUDA_FP16, OPENCL, OPENCL_FP16, VULKAN };

struct BackendSelection {
    ComputeDevice device = ComputeDevice::CPU;
    std::string reason;
};

class ONNXSegmenter {
  public:
    explicit ONNXSegmenter(const Config &config);

    [[nodiscard]] const Config &config() const noexcept;
    [[nodiscard]] const std::map<int, std::string> &class_map() const noexcept;

    [[nodiscard]] SegmentationResult
    predict(const std::string &image_path) const;
    [[nodiscard]] SegmentationResult predict(const cv::Mat &image_bgr) const;

  private:
    Config config_;
    cv::dnn::Net net_;
    std::map<int, std::string> id_to_label;
    std::map<int, cv::Vec3b> colors_;

    void validate_config() const;
    void load_model(const std::string &model_path);
    void configure_backend();
    BackendSelection choose_best_backend() const;
    void load_class_map(const std::string &class_map_path);
    void build_colors();

    bool has_target(const std::vector<cv::dnn::Target> &targets,
                    cv::dnn::Target t) const;

    [[nodiscard]] cv::Mat make_blob(const cv::Mat &image_bgr) const;
    [[nodiscard]] cv::Mat run_inference(const cv::Mat &blob) const;
    [[nodiscard]] cv::Mat decode_class_mask(const cv::Mat &output) const;
    [[nodiscard]] cv::Mat
    resize_mask_to_original(const cv::Mat &mask,
                            const cv::Size &original_size) const;
    [[nodiscard]] SegmentationResult compose_result(const cv::Mat &image_bgr,
                                                    const cv::Mat &mask) const;
};

}