cmake_minimum_required(VERSION 3.10)

project(2DFilter)

set(CMAKE_CUDA_COMPILER "/usr/bin/nvcc")
find_package(CUDA 9.1 REQUIRED)
find_package(OpenCV REQUIRED)

project(xact-build LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4 -std=c++17 -Wall -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)
include_directories("${CUDA_INCLUDE_DIRS}")
add_executable(
  2DFilter
  source/filter.cu
  source/main.cpp
  include/cuda_timer.h
  include/filter.h
  include/timer.h)

target_link_libraries(2DFilter ${OpenCV_LIBS})