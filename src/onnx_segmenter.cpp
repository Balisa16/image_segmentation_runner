#include "segmenter/onnx_segmenter.hpp"
#include "segmenter/errors.hpp"

#include <algorithm>
#include <cfloat>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <regex>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace Segmenter {

[[nodiscard]] bool file_exists(const std::string &path) {
    std::error_code ec;
    return !path.empty() && fs::exists(path, ec) &&
           fs::is_regular_file(path, ec);
}

const char *to_string(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::InvalidConfig:
        return "InvalidConfig";
    case ErrorCode::FileNotFound:
        return "FileNotFound";
    case ErrorCode::ModelLoadFailed:
        return "ModelLoadFailed";
    case ErrorCode::ClassMapOpenFailed:
        return "ClassMapOpenFailed";
    case ErrorCode::ClassMapParseFailed:
        return "ClassMapParseFailed";
    case ErrorCode::ImageLoadFailed:
        return "ImageLoadFailed";
    case ErrorCode::InvalidModelOutput:
        return "InvalidModelOutput";
    case ErrorCode::InferenceFailed:
        return "InferenceFailed";
    case ErrorCode::UnsupportedBatchSize:
        return "UnsupportedBatchSize";
    case ErrorCode::InternalError:
        return "InternalError";
    default:
        return "Unknown";
    }
}

ONNXSegmenter::ONNXSegmenter(const Config &config) : config_(config) {
    validate_config();
    load_class_map(config_.class_map_path);
    load_model(config_.model_path);
    build_colors();
}

const Config &ONNXSegmenter::config() const noexcept { return config_; }

const std::map<int, std::string> &ONNXSegmenter::class_map() const noexcept {
    return id_to_label;
}

void ONNXSegmenter::validate_config() const {
    if (!config_.is_valid()) {
        std::ostringstream oss;
        oss << "Invalid configuration:"
            << " model_path='" << config_.model_path << "'"
            << ", class_map_path='" << config_.class_map_path << "'"
            << ", image_size=" << config_.image_size
            << ", alpha=" << config_.alpha
            << ", min_pixels=" << config_.min_pixels;
        throw SegmenterException(ErrorCode::InvalidConfig, oss.str());
    }

    if (!file_exists(config_.model_path))
        throw SegmenterException(ErrorCode::FileNotFound,
                                 "Model file not found: " + config_.model_path);

    if (!file_exists(config_.class_map_path))
        throw SegmenterException(ErrorCode::FileNotFound,
                                 "Class map file not found: " +
                                     config_.class_map_path);
}

bool ONNXSegmenter::has_target(const std::vector<cv::dnn::Target> &targets,
                               cv::dnn::Target t) const {
    return std::find(targets.begin(), targets.end(), t) != targets.end();
}

BackendSelection ONNXSegmenter::choose_best_backend() const {
    BackendSelection result{ComputeDevice::CPU, "fallback to CPU"};

    const auto cuda_targets =
        cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_CUDA);
    const auto opencv_targets =
        cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_OPENCV);
    const auto vk_targets =
        cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_VKCOM);

    // 1. NVIDIA CUDA
    const int cuda_device_count = cv::cuda::getCudaEnabledDeviceCount();
    if (cuda_device_count > 0 &&
        has_target(cuda_targets, cv::dnn::DNN_TARGET_CUDA_FP16))
        return {ComputeDevice::CUDA_FP16, "NVIDIA CUDA FP16 available"};

    if (cuda_device_count > 0 &&
        has_target(cuda_targets, cv::dnn::DNN_TARGET_CUDA))
        return {ComputeDevice::CUDA, "NVIDIA CUDA available"};

    // Generic OpenCL (AMD / Intel)
    if (cv::ocl::haveOpenCL()) {
        cv::ocl::setUseOpenCL(true);
        if (cv::ocl::useOpenCL()) {
            if (has_target(opencv_targets, cv::dnn::DNN_TARGET_OPENCL_FP16))
                return {ComputeDevice::OPENCL_FP16, "OpenCL FP16 available"};
            if (has_target(opencv_targets, cv::dnn::DNN_TARGET_OPENCL))
                return {ComputeDevice::OPENCL, "OpenCL available"};
        }
    }

    if (has_target(vk_targets, cv::dnn::DNN_TARGET_VULKAN))
        return {ComputeDevice::VULKAN, "Vulkan target available"};

    return result;
}

void ONNXSegmenter::configure_backend() {
    const BackendSelection choice = choose_best_backend();

    try {
        switch (choice.device) {
        case ComputeDevice::CUDA_FP16:
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
            break;

        case ComputeDevice::CUDA:
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            break;

        case ComputeDevice::OPENCL_FP16:
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);
            break;

        case ComputeDevice::OPENCL:
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
            break;

        case ComputeDevice::VULKAN:
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_VKCOM);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_VULKAN);
            break;

        case ComputeDevice::CPU:
        default:
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            break;
        }

#ifdef DEBUG_MODE
        std::cout << "DNN backend selected: " << choice.reason << '\n';
#endif
    } catch (const cv::Exception &ex) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

#ifdef DEBUG_MODE
        std::cout << "GPU backend selection failed, using CPU: " << ex.what()
                  << '\n';
#endif
    }
}

void ONNXSegmenter::load_model(const std::string &model_path) {
    try {
        net_ = cv::dnn::readNetFromONNX(model_path);
    } catch (const cv::Exception &ex) {
        throw SegmenterException(ErrorCode::ModelLoadFailed,
                                 "OpenCV failed to load ONNX model '" +
                                     model_path + "': " + ex.what());
    }

    if (net_.empty())
        throw SegmenterException(ErrorCode::ModelLoadFailed,
                                 "Loaded network is empty for model: " +
                                     model_path);

    configure_backend();
}

void ONNXSegmenter::load_class_map(const std::string &class_map_path) {
    std::ifstream file(class_map_path);
    if (!file.is_open()) {
        throw SegmenterException(ErrorCode::ClassMapOpenFailed,
                                 "Failed to open class map: " + class_map_path);
    }

    const std::string content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());

    const std::regex pairRegex(R"regex("([^"]+)"\s*:\s*(\d+))regex");

    id_to_label.clear();

    auto begin =
        std::sregex_iterator(content.begin(), content.end(), pairRegex);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        const std::string label = (*it)[1].str();
        const int id = std::stoi((*it)[2].str());
        id_to_label[id] = label;
    }

    if (id_to_label.empty())
        throw SegmenterException(
            ErrorCode::ClassMapParseFailed,
            "Class map JSON is empty or unsupported. Expected format like "
            R"({"background":0,"defect":1})");

    // if (!id_to_label.contains(0))
}

void ONNXSegmenter::build_colors() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);

    colors_.clear();
    for (const auto &[class_id, label] : id_to_label) {
        (void)label;
        colors_[class_id] = cv::Vec3b(static_cast<uchar>(dist(rng)),
                                      static_cast<uchar>(dist(rng)),
                                      static_cast<uchar>(dist(rng)));
    }
}

cv::Mat ONNXSegmenter::make_blob(const cv::Mat &image_bgr) const {
    if (image_bgr.empty())
        throw SegmenterException(ErrorCode::ImageLoadFailed,
                                 "Input image is empty.");

    cv::Mat image_rgb;
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);

    return cv::dnn::blobFromImage(
        image_rgb, 1.0 / 255.0,
        cv::Size(config_.image_size, config_.image_size), cv::Scalar(0, 0, 0),
        true, false);
}

cv::Mat ONNXSegmenter::run_inference(const cv::Mat &blob) const {
    try {
        cv::dnn::Net net_copy = net_;
        net_copy.setInput(blob);
        return net_copy.forward();
    } catch (const cv::Exception &ex) {
        throw SegmenterException(ErrorCode::InferenceFailed,
                                 "Inference failed: " + std::string(ex.what()));
    }
}

cv::Mat ONNXSegmenter::decode_class_mask(const cv::Mat &output) const {
    if (output.dims != 4)
        throw SegmenterException(
            ErrorCode::InvalidModelOutput,
            "Unexpected output dims: " + std::to_string(output.dims) +
                ". Expected 4D tensor [N, C, H, W].");

    const int batch = output.size[0];
    const int num_classes = output.size[1];
    const int outH = output.size[2];
    const int outW = output.size[3];

    if (batch != 1)
        throw SegmenterException(
            ErrorCode::UnsupportedBatchSize,
            "Only batch size 1 is supported. Actual batch size: " +
                std::to_string(batch));

    if (num_classes <= 0 || outH <= 0 || outW <= 0)
        throw SegmenterException(ErrorCode::InvalidModelOutput,
                                 "Invalid output shape from model.");

    cv::Mat class_mask(outH, outW, CV_8U, cv::Scalar(0));

    for (int y = 0; y < outH; ++y)
        for (int x = 0; x < outW; ++x) {
            float best_score = -FLT_MAX;
            int best_class = 0;

            for (int c = 0; c < num_classes; ++c) {
                const int idx[4] = {0, c, y, x};
                const float score = output.at<float>(idx);

                if (score > best_score) {
                    best_score = score;
                    best_class = c;
                }
            }

            class_mask.at<uchar>(y, x) = static_cast<uchar>(best_class);
        }

    return class_mask;
}

cv::Mat
ONNXSegmenter::resize_mask_to_original(const cv::Mat &mask,
                                       const cv::Size &original_size) const {
    cv::Mat resized;
    cv::resize(mask, resized, original_size, 0, 0, cv::INTER_NEAREST);
    return resized;
}

SegmentationResult ONNXSegmenter::compose_result(const cv::Mat &image_bgr,
                                                 const cv::Mat &mask) const {
    if (image_bgr.empty() || mask.empty())
        throw SegmenterException(
            ErrorCode::InternalError,
            "Cannot compose result from empty image or empty mask.");

    if (image_bgr.rows != mask.rows || image_bgr.cols != mask.cols)
        throw SegmenterException(ErrorCode::InternalError,
                                 "Image and mask sizes do not match.");

    cv::Mat overlay = image_bgr.clone();
    std::map<std::string, int> detected_classes;

    for (const auto &[class_id, label] : id_to_label) {
        if (class_id == 0)
            continue;

        cv::Mat binary = (mask == class_id);
        const int pixel_count = cv::countNonZero(binary);

        if (pixel_count < config_.min_pixels)
            continue;

        detected_classes[label] = pixel_count;

        const auto colorIt = colors_.find(class_id);
        const cv::Vec3b color =
            (colorIt != colors_.end()) ? colorIt->second : cv::Vec3b(0, 255, 0);

        for (int y = 0; y < overlay.rows; ++y)
            for (int x = 0; x < overlay.cols; ++x)
                if (binary.at<uchar>(y, x)) {
                    cv::Vec3b &px = overlay.at<cv::Vec3b>(y, x);
                    for (int k = 0; k < 3; ++k)
                        px[k] =
                            static_cast<uchar>((1.0f - config_.alpha) * px[k] +
                                               config_.alpha * color[k]);
                }
    }

    return SegmentationResult{.overlay_bgr = overlay,
                              .class_mask = mask.clone(),
                              .detected_classes = std::move(detected_classes)};
}

SegmentationResult ONNXSegmenter::predict(const std::string &image_path) const {
    cv::Mat image_bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image_bgr.empty())
        throw SegmenterException(ErrorCode::ImageLoadFailed,
                                 "Failed to load image: " + image_path);

    return predict(image_bgr);
}

SegmentationResult ONNXSegmenter::predict(const cv::Mat &image_bgr) const {
    const cv::Size original_size(image_bgr.cols, image_bgr.rows);

#ifdef DEBUG_MODE
    auto dt_start = std::chrono::high_resolution_clock::now();
#endif
    const cv::Mat blob = make_blob(image_bgr);
    const cv::Mat output = run_inference(blob);
    const cv::Mat small_mask = decode_class_mask(output);
    const cv::Mat full_mask =
        resize_mask_to_original(small_mask, original_size);
#ifdef DEBUG_MODE
    std::cout << "\033[1;32mPrediction time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - dt_start)
                     .count()
              << "ms\033[0m\n";
#endif

    return compose_result(image_bgr, full_mask);
}

} // namespace Segmenter