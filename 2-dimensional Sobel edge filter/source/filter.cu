#include "cuda_timer.h"
#include "filter.h"

#include <iostream>

using namespace std;

__global__ void kernel_sobel_filter(const uchar *input, uchar *output,
                                    const uint height, const uint width)
{

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  // const int sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  // const int sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  const int sobel_x[FILTER_DIM][FILTER_DIM] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int sobel_y[FILTER_DIM][FILTER_DIM] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  const int tid = tx + blockDim.x * ty;
  // Loading of the tiles
  int row = by * OUTPUT_TILEDIM + ty - FILTER_RADIUS;
  int col = bx * OUTPUT_TILEDIM + tx - FILTER_RADIUS;

  extern __shared__ int sh_input[];
  float pixel_x = 0;
  float pixel_y = 0;

  if (row >= 0 && row < height && col >= 0 && col < width)
  {
    sh_input[tid] = input[col + row * width];
  }
  else
    sh_input[tid] = 0;

  __syncthreads();
  // Convolution operation
  if (row >= 0 && row < height && col >= 0 && col < width)
  {

    const int rowTile = ty - FILTER_RADIUS;
    const int colTile = tx - FILTER_RADIUS;
    if (rowTile >= 0 && rowTile < OUTPUT_TILEDIM && colTile >= 0 &&
        colTile < OUTPUT_TILEDIM)
    {

      for (int i = 0; i < FILTER_DIM; i++)
      {
        for (int j = 0; j < FILTER_DIM; j++)
        {
          pixel_x += (sobel_x[i][j] * sh_input[(tx - 1 + j) + (ty - 1 + i) * BLOCKDIM]);
        }
      }

      for (int i = 0; i < FILTER_DIM; i++)
      {
        for (int j = 0; j < FILTER_DIM; j++)
        {
          pixel_y += (sobel_y[i][j] * sh_input[(tx - 1 + j) + (ty - 1 + i) * BLOCKDIM]);
        }
      }

      pixel_x *= pixel_x;
      pixel_y *= pixel_y;

      int magnitude = sqrt((float)(pixel_x + pixel_y));

      if (magnitude < 0)
      {
        magnitude = 0;
      }
      if (magnitude > 255)
      {
        magnitude = 255;
      }

      output[col + row * width] = (int)magnitude;
    }
  }
}

inline int divup(int a, int b)
{
  if (a % b)
    return a / b + 1;
  else
    return a / b;
}

/**
 * Wrapper for calling the kernel.
 */
void sobel_filter_gpu(const uchar *input, uchar *output, const uint height,
                      const uint width)
{
  const int size = height * width * sizeof(uchar);

  CudaSynchronizedTimer timer;

  // Launch the kernel
  const int grid_x = width / OUTPUT_TILEDIM;
  const int grid_y = height / OUTPUT_TILEDIM;

  dim3 grid(grid_x, grid_y, 1);      // TODO
  dim3 block(BLOCKDIM, BLOCKDIM, 1); // TODO

  timer.start();
  kernel_sobel_filter<<<grid, block, 48 * 1024>>>(input, output, height, width);
  timer.stop();

  cudaDeviceSynchronize();

  float time_kernel = timer.getElapsed();
}

void sobel_filter_cpu(const uchar *input, uchar *output, const uint height,
                      const uint width)
{
  const int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  for (uint y = 1; y < height - 1; ++y)
  {
    for (uint x = 1; x < width - 1; ++x)
    {

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

      if (magnitude < 0)
      {
        magnitude = 0;
      }
      if (magnitude > 255)
      {
        magnitude = 255;
      }

      output[x + y * width] = magnitude;
    }
  }
}
