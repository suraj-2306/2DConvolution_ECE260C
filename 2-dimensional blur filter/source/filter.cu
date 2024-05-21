#include "cuda_timer.h"
#include "filter.h"

#include <iostream>

using namespace std;

__global__ void kernel_filter(const uchar *input, uchar *output,
                              const uint height, const uint width)
{

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const float filter[FILTER_DIM][FILTER_DIM] = {
      // {0.0625, 0.125, 0.0625}, {0.125, 0.25, 0.125}, {0.0625, 0.125, 0.0625}};
      {0.25, 0, 0, 0, 0.25},
      {0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0},
      {0.25, 0, 0, 0, 0.25}};

  const int tid = tx + blockDim.x * ty;
  // Loading of the tiles
  int row = by * OUTPUT_TILEDIM + ty - FILTER_RADIUS;
  int col = bx * OUTPUT_TILEDIM + tx - FILTER_RADIUS;

  extern __shared__ float sh_input[];
  float pixel = 0;

  if (row >= 0 && row < height && col >= 0 && col < width)
  {
    sh_input[tid] = (float)input[col + row * width];
  }
  else
    sh_input[tid] = 0;

  __syncthreads();
  // Convolution operation
  if (row >= 0 && row < height && col >= 0 && col < width)
  {

    const int rowTile = ty - FILTER_RADIUS;
    const int colTile = tx - FILTER_RADIUS;
    if (rowTile >= 0 && rowTile < OUTPUT_TILEDIM && colTile >= 0 && colTile < OUTPUT_TILEDIM)
    {

      for (int i = 0; i < FILTER_DIM; i++)
      {
        for (int j = 0; j < FILTER_DIM; j++)
        {
          pixel += (filter[i][j] * sh_input[(colTile + j) + (rowTile + i) * BLOCKDIM]);
        }
      }

      if (pixel < 0)
        pixel = 0;
      if (pixel > 255)
        pixel = 255;

      output[col + row * width] = (int)pixel;
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
void filter_gpu(const uchar *input, uchar *output, const uint height,
                const uint width)
{
  const int size = height * width * sizeof(uchar);

  CudaSynchronizedTimer timer;

  // Launch the kernel
  int grid_x = width / OUTPUT_TILEDIM;
  int grid_y = height / OUTPUT_TILEDIM;

  if (width % OUTPUT_TILEDIM)
    grid_x++;
  if (height % OUTPUT_TILEDIM)
    grid_y++;

  dim3 grid(grid_x, grid_y, 1);
  dim3 block(BLOCKDIM, BLOCKDIM, 1);

  timer.start();
  kernel_filter<<<grid, block, 48 * 1024>>>(input, output, height, width);
  timer.stop();

  cudaDeviceSynchronize();

  float time_kernel = timer.getElapsed();
}

void filter_cpu(const uchar *input, uchar *output, const uint height,
                const uint width)
{
  const float filter[FILTER_DIM][FILTER_DIM] = {
      // {0.0625, 0.125, 0.0625}, {0.125, 0.25, 0.125}, {0.0625, 0.125, 0.0625}};
      {0.25, 0, 0, 0, 0.25},
      {0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0},
      {0.25, 0, 0, 0, 0.25}};

  for (uint y = FILTER_RADIUS; y < height - FILTER_RADIUS; ++y)
  {
    for (uint x = FILTER_RADIUS; x < width - FILTER_RADIUS; ++x)
    {
      float pixel = 0;
      for (int i = 0; i < FILTER_DIM; i++)
      {
        for (int j = 0; j < FILTER_DIM; j++)
        {
          pixel += (float)(filter[i][j] * input[(x - FILTER_RADIUS + j) + (y - FILTER_RADIUS + i) * width]);
        }
      }

      if (pixel < 0)
        pixel = 0;
      if (pixel > 255)
        pixel = 255;

      output[x + y * width] = (int)pixel;
    }
  }
}