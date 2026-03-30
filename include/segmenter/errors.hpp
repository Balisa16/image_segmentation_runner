#pragma once

#include <stdexcept>
#include <string>

namespace Segmenter {

enum class ErrorCode {
    InvalidConfig,
    FileNotFound,
    ModelLoadFailed,
    ClassMapOpenFailed,
    ClassMapParseFailed,
    ImageLoadFailed,
    InvalidModelOutput,
    InferenceFailed,
    UnsupportedBatchSize,
    InternalError,
    EmptyImage
};

class SegmenterException : public std::runtime_error {
  public:
    SegmenterException(ErrorCode code, std::string message)
        : std::runtime_error(std::move(message)), code_(code) {}

    [[nodiscard]] ErrorCode code() const noexcept { return code_; }

  private:
    ErrorCode code_;
};

[[nodiscard]] const char *to_string(ErrorCode code) noexcept;

}