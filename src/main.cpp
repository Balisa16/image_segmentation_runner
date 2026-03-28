#define DEBUG_MODE
#include "segmenter/errors.hpp"
#include "segmenter/onnx_segmenter.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace Segmenter;

void print_usage(const char *prog_name) {
    std::cout << "Usage:\n"
              << "  " << prog_name
              << " <model.onnx> <model.json> <image> [image_size] [alpha] "
                 "[min_pixels]\n\n"
              << "Example:\n"
              << "  " << prog_name
              << " models/model.onnx models/model.json sample.jpg 512 0.5 50\n";
}

Config parse_args(int argc, char **argv, std::string &image_path) {
    if (argc < 4)
        throw SegmenterException(ErrorCode::InvalidConfig,
                                 "Not enough arguments.");

    Config cfg;
    cfg.model_path = argv[1];
    cfg.class_map_path = argv[2];
    image_path = argv[3];

    try {
        if (argc >= 5)
            cfg.image_size = std::stoi(argv[4]);
        if (argc >= 6)
            cfg.alpha = std::stof(argv[5]);
        if (argc >= 7)
            cfg.min_pixels = std::stoi(argv[6]);
    } catch (const std::exception &) {
        throw SegmenterException(ErrorCode::InvalidConfig,
                                 "Failed to parse optional arguments: "
                                 "image_size, alpha, or min_pixels.");
    }

    if (!cfg.is_valid())
        throw SegmenterException(ErrorCode::InvalidConfig,
                                 "Parsed configuration is invalid.");

    return cfg;
}

void print_detected_classes(const std::map<std::string, int> &detected) {
    std::cout << "Detected classes:\n";
    if (detected.empty()) {
        std::cout << "  (none)\n";
        return;
    }

    for (const auto &[label, pixels] : detected)
        std::cout << "  - " << label << ": " << pixels << " pixels\n";
}

int main(int argc, char **argv) {
    try {
        if (argc < 4) {
            print_usage(argv[0]);
            return 1;
        }

        std::string image_path;
        const Config config = parse_args(argc, argv, image_path);

        const ONNXSegmenter segmenter(config);

        const SegmentationResult result = segmenter.predict(image_path);

        print_detected_classes(result.detected_classes);

        cv::imshow("Segmentation", result.overlay_bgr);
        cv::waitKey(0);
        cv::destroyAllWindows();

        return 0;
    } catch (const SegmenterException &ex) {
        std::cerr << "[SegmenterError][" << to_string(ex.code()) << "] "
                  << ex.what() << '\n';
        return 1;
    } catch (const cv::Exception &ex) {
        std::cerr << "[OpenCVError] " << ex.what() << '\n';
        return 2;
    } catch (const std::exception &ex) {
        std::cerr << "[StdError] " << ex.what() << '\n';
        return 3;
    } catch (...) {
        std::cerr << "[UnknownError] Unexpected failure.\n";
        return 99;
    }
}