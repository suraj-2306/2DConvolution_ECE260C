cmake_minimum_required(VERSION 3.10)

project(1DFilter)

set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")
find_package(CUDA 12.4 REQUIRED)

project(xact-build LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4 -std=c++17 -Wall -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75 --ptxas-options=-v")

include_directories(include)
include_directories("${CUDA_INCLUDE_DIRS}")
add_executable(
  1DFilter
  source/fir_gpu.cu
  source/main.cpp
  include/cuda_error.h
  include/cuda_timer.h
  include/fir_gpu.h
  include/main.h
  include/timer.h)
