#include "cuda_timer.h"
#include "filter.h"

#include <iostream>

using namespace std;

__global__ void kernel_sobel_filter(const uchar *input, uchar *output,
                                    const uint height, const uint width) {
  // TODO

  const int tx = threadIdx.x;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  const int sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  extern __shared__ int sh_input[];
  const int tidx = tx % 3, tidy = tx / 3;
  int pixelx = 0;
  int pixely = 0;
  if ((bx > 1 && bx < width) && (by > 0 && by < height)) {
    input += (bx - 1) + (by - 1) * width;

    sh_input[tx] = input[tidx + tidy * width];

    sh_input[tx] *= sobel_x[tx];
    __syncthreads();

    // if (tx == 0) {
    for (int i = 0; i < 9; i++)
      pixelx += sh_input[i];
    pixelx *= pixelx;
    // }

    sh_input[tx] = input[tidx + tidy * width];

    sh_input[tx] *= sobel_y[tx];
    __syncthreads();

    for (int i = 0; i < 9; i++)
      pixely += sh_input[i];
    pixely *= pixely;

    if (tx == 0)
      output[by * width + bx] = sqrt((float)(pixelx + pixely));
  } else {
    if (tx == 0)
      output[by * width + bx] = input[bx + by * width];
  }
}

inline int divup(int a, int b) {
  if (a % b)
    return a / b + 1;
  else
    return a / b;
}

/**
 * Wrapper for calling the kernel.
 */
void sobel_filter_gpu(const uchar *input, uchar *output, const uint height,
                      const uint width) {
  const int size = height * width * sizeof(uchar);

  CudaSynchronizedTimer timer;

  // Launch the kernel
  const int grid_x = 64;
  const int grid_y = 64;

  dim3 grid(height, width, 1); // TODO
  dim3 block(9, 1, 1);         // TODO

  timer.start();
  kernel_sobel_filter<<<grid, block, 48 * 1024>>>(input, output, height, width);
  timer.stop();

  cudaDeviceSynchronize();

  float time_kernel = timer.getElapsed();
}

void sobel_filter_cpu(const uchar *input, uchar *output, const uint height,
                      const uint width) {
  const int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  for (uint y = 1; y < height - 1; ++y) {
    for (uint x = 1; x < width - 1; ++x) {

      const int pixel_x =
          (int)((sobel_x[0][0] * input[x - 1 + (y - 1) * width]) +
                (sobel_x[0][1] * input[x + (y - 1) * width]) +
                (sobel_x[0][2] * input[x + 1 + (y - 1) * width]) +
                (sobel_x[1][0] * input[x - 1 + (y)*width]) +
                (sobel_x[1][1] * input[x + (y)*width]) +
                (sobel_x[1][2] * input[x + 1 + (y)*width]) +
                (sobel_x[2][0] * input[x - 1 + (y + 1) * width]) +
                (sobel_x[2][1] * input[x + (y + 1) * width]) +
                (sobel_x[2][2] * input[x + 1 + (y + 1) * width]));
      const int pixel_y =
          (int)((sobel_y[0][0] * input[x - 1 + (y - 1) * width]) +
                (sobel_y[0][1] * input[x + (y - 1) * width]) +
                (sobel_y[0][2] * input[x + 1 + (y - 1) * width]) +
                (sobel_y[1][0] * input[x - 1 + (y)*width]) +
                (sobel_y[1][1] * input[x + (y)*width]) +
                (sobel_y[1][2] * input[x + 1 + (y)*width]) +
                (sobel_y[2][0] * input[x - 1 + (y + 1) * width]) +
                (sobel_y[2][1] * input[x + (y + 1) * width]) +
                (sobel_y[2][2] * input[x + 1 + (y + 1) * width]));

      float magnitude = sqrt((float)(pixel_x * pixel_x + pixel_y * pixel_y));

      if (magnitude < 0) {
        magnitude = 0;
      }
      if (magnitude > 255) {
        magnitude = 255;
      }

      output[x + y * width] = magnitude;
    }
  }
}
