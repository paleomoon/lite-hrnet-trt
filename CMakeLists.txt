cmake_minimum_required(VERSION 3.10)
project(lite-hrnet-trt)

set(OpenCV_DIR "D:/opencv344/opencv/build/x64/vc15/lib")
find_package(OpenCV  REQUIRED core imgproc)
if(OpenCV_FOUND)
    message(STATUS "OpenCV FOUND")
endif()

include_directories("D:/software/TensorRT-7.2.2.3/include" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include")

add_executable(${PROJECT_NAME} src/LiteHRNet.cpp)

file(GLOB trt-libs "D:/software/TensorRT-7.2.2.3/lib/*.lib")
file(GLOB cuda-libs "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/lib/x64/*.lib")

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${trt-libs} ${cuda-libs})