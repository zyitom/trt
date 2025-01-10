cmake_minimum_required(VERSION 3.10)
project(test_trt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA REQUIRED)


file(GLOB COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB UTILS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/.cpp")

find_package(OpenCV REQUIRED)

message(STATUS "Common sources: ${COMMON_SOURCES}")
message(STATUS "Utils sources: ${UTILS_SOURCES}")

link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib
    /usr/lib/wsl/lib
    /usr/local/lib
    /usr/lib/x86_64-linux-gnu
    ${OpenCV_LIBS}
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    /usr/include/x86_64-linux-gnu
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${OpenCV_INCLUDE_DIRS}
)


add_executable(trt_exec
    trtexec.cpp
    ${COMMON_SOURCES}
    ${UTILS_SOURCES}
        sampleUtils.cpp
)

target_link_libraries(trt_exec
    -L${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib
    -L/usr/lib/wsl/lib
    -L/usr/local/lib
    -L/usr/lib/x86_64-linux-gnu
    -lcuda
    -lcudart
    -lnvinfer
    -lnvonnxparser
    -lnvinfer_plugin
    -ldl
    -lpthread
)
