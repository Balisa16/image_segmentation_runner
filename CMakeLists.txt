cmake_minimum_required(VERSION 3.16)
project(aoi_segmenter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_library(image_segmenter
    src/onnx_segmenter.cpp
)

target_include_directories(image_segmenter
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_compile_definitions(image_segmenter PRIVATE
    $<$<CONFIG:Debug>:DEBUG_MODE>
)

target_link_libraries(image_segmenter
    PUBLIC
        ${OpenCV_LIBS}
)

add_executable(aoi_segmenter
    src/main.cpp
)

target_compile_definitions(aoi_segmenter PRIVATE
    $<$<CONFIG:Debug>:DEBUG_MODE>
)

target_link_libraries(aoi_segmenter
    PRIVATE
        image_segmenter
)