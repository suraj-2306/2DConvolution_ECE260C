
#include "cuda_timer.h"
#include "fir_gpu.h"

#include <iostream>

#define BLOCK_SIZE 64

// Baseline
__global__ void fir_kernel1(const float *coeffs, const float *input,
                            float *output, int length, int filterLength) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;

  input += bx * BLOCK_SIZE;
  output += bx * BLOCK_SIZE;
  float acc = 0;
  for (int n = 0; n < filterLength; n++) {
    acc += coeffs[n] * input[n + tx];
    output[tx] = acc;
  }
}

// Coefficients in shared memory
// Here we suppose that filterLength and BLOCK_SIZE is always 64
__global__ void fir_kernel2(const float *coeffs, const float *input,
                            float *output, int length, int filterLength) {

  extern __shared__ float sh_coeffs[];

  int tx = threadIdx.x;
  int bx = blockIdx.x;

  sh_coeffs[tx] = coeffs[tx];
  __syncthreads();

  input += bx * BLOCK_SIZE;
  output += bx * BLOCK_SIZE;
  float acc = 0;
  for (int n = 0; n < filterLength; n++) {
    acc += sh_coeffs[n] * input[n + tx];
  }

  output[tx] = acc;
}

// Coefficients and inputs in shared memory
// Here we suppose that filterLength and BLOCK_SIZE is always 64
__global__ void fir_kernel3(const float *coeffs, const float *input,
                            float *output, int length, int filterLength) {
  extern __shared__ float combinedMem[];
  float *sh_coeffs = combinedMem;
  float *sh_input = combinedMem + BLOCK_SIZE;

  int tx = threadIdx.x;
  int bx = blockIdx.x;

  input += bx * BLOCK_SIZE;
  sh_coeffs[tx] = coeffs[tx];
  sh_input[tx] = input[tx];
  sh_input[tx + BLOCK_SIZE] = input[tx + BLOCK_SIZE];
  __syncthreads();

  output += bx * BLOCK_SIZE;
  float acc = 0;
  for (int n = 0; n < filterLength; n++) {
    acc += sh_coeffs[n] * sh_input[n + tx];
  }
  output[tx] = acc;
}

inline int divup(int a, int b) {
  if (a % b)
    return a / b + 1;
  else
    return a / b;
}

void fir_gpu(const float *coeffs, const float *input, float *output, int length,
             int filterLength) {
  const int output_size = length - filterLength;

  CudaSynchronizedTimer timer;

  const int block_size = BLOCK_SIZE;

  dim3 block(block_size, 1, 1);

  int numblocks = length / block_size;

  if (length % numblocks)
    numblocks++;

  dim3 grid(numblocks, 1, 1);

  timer.start();
  fir_kernel3<<<grid, block, 1024 * 48>>>(coeffs, input, output, length,
                                          filterLength);
  // TODO Launch kernel here
  timer.stop();

  cudaDeviceSynchronize();

  CudaCheckError();

  float time_gpu = timer.getElapsed();

  // std::cout << "Kernel Time: " << time_gpu << "ms\n";
}
