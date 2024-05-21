#include <cmath>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "filter.h"
#include "timer.h"

using namespace std;
using namespace cv;

enum filterType
{
  BLUR_CPU,
  BLUR_GPU,
  BLUR_GPU_UNI
};

int main(int argc, const char *argv[])
{
  VideoCapture cap("/mnt/c/Users/ashwi/Desktop/Ashwin/UCSD/Quarters/SP24/ECE260C/Lab3/git/260CCUDA/2-dimensional blur filter/source/input.raw");
  // VideoCapture cap(1); //output of ls /dev/video* will display available
  // video devices and their indices

  int WIDTH = 768;
  int HEIGHT = 768;

  filterType filter_type = BLUR_GPU;

  // 1 argument on command line: WIDTH = HEIGHT = arg
  if (argc >= 2)
  {
    WIDTH = atoi(argv[1]);
    HEIGHT = WIDTH;
  }
  // 2 arguments on command line: WIDTH = arg1, HEIGHT = arg2
  if (argc >= 3)
  {
    HEIGHT = atoi(argv[2]);
  }

  if (argc >= 4)
  {
    filter_type = static_cast<filterType>(atoi(argv[3]));
  }

  switch (filter_type)
  {
  case BLUR_CPU:
    cout << "Using CPU" << endl;
    break;
  case BLUR_GPU:
    cout << "Using GPU" << endl;
    break;
  case BLUR_GPU_UNI:
    cout << "Using GPU with Unified Memory" << endl;
    break;
  }

  // Profiling
  LinuxTimer timer;
  LinuxTimer fps_counter;
  double time_elapsed = 0;

  // Allocate memory
  unsigned char *gray_ptr;
  unsigned char *out_ptr;
  unsigned char *gray_uni_ptr;
  unsigned char *out_uni_ptr;
  uchar *d_grayPtr, *d_outPtr;

  // Allocate memory
  gray_ptr = (unsigned char *)malloc(WIDTH * HEIGHT * sizeof(unsigned char));
  out_ptr = (unsigned char *)malloc(WIDTH * HEIGHT * sizeof(unsigned char));
  cudaMallocManaged(&gray_uni_ptr, WIDTH * HEIGHT * sizeof(unsigned char));
  cudaMallocManaged(&out_uni_ptr, WIDTH * HEIGHT * sizeof(unsigned char));

  cudaError_t err;
  err = cudaMalloc(&d_outPtr, WIDTH * HEIGHT * sizeof(uchar));
  if (err != cudaSuccess)
  {
    fprintf(stderr, "GPU_ERROR: cudaMalloc failed output!\n");
    exit(1);
  }

  err = cudaMalloc(&d_grayPtr, WIDTH * HEIGHT * sizeof(uchar));
  if (err != cudaSuccess)
  {
    fprintf(stderr, "GPU_ERROR: cudaMalloc failed grayptr!\n");
    exit(1);
  }

  // Mat gray = Mat(HEIGHT, WIDTH, CV_8U, gray_ptr);
  // Mat out = Mat(HEIGHT, WIDTH, CV_8U, out_ptr);
  Mat gray;
  Mat out;
  switch (filter_type)
  {
  case BLUR_CPU:
  case BLUR_GPU:
    gray = Mat(HEIGHT, WIDTH, CV_8U, gray_ptr);
    out = Mat(HEIGHT, WIDTH, CV_8U, out_ptr);
    break;

  case BLUR_GPU_UNI:
    gray = Mat(HEIGHT, WIDTH, CV_8U, gray_uni_ptr);
    out = Mat(HEIGHT, WIDTH, CV_8U, out_uni_ptr);
    break;
  }

  // More declarations
  Mat frame;

  char key = 0;
  int count = 0;

  // Main loop
  while (key != 'q')
  {
    // Get frame
    cap >> frame;

    // If no more frames, wait and exit
    if (frame.empty())
    {
      waitKey();
      break;
    }

    // Resize and grayscale
    resize(frame, frame, Size(WIDTH, HEIGHT));
    cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Transfer to GPU memory
    if (filter_type == BLUR_GPU)
    {
      err = cudaMemcpy(d_grayPtr, gray.ptr<uchar>(),
                       WIDTH * HEIGHT * sizeof(uchar), cudaMemcpyHostToDevice);

      if (err != cudaSuccess)
      {
        fprintf(stderr, "GPU_ERROR: Host to device cudaMemCpy failed for grayptr!\n");
        exit(1);
      }
    }

    // Run filter
    timer.start();
    out = (Scalar)0;

    switch (filter_type)
    {
    case BLUR_CPU:
      filter_cpu(gray.ptr<uchar>(), out.ptr<uchar>(), gray.rows, gray.cols);
      break;
    case BLUR_GPU:
      filter_gpu(d_grayPtr, d_outPtr, gray.rows, gray.cols);
      err = cudaMemcpy(out.ptr<uchar>(), d_outPtr,
                       WIDTH * HEIGHT * sizeof(uchar), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess)
      {
        fprintf(stderr, "GPU_ERROR: Device to host cudaMemCpy failed for output!\n");
        exit(1);
      }
      break;
    case BLUR_GPU_UNI:
      filter_gpu(gray.ptr<uchar>(), out.ptr<uchar>(), gray.rows, gray.cols);
      break;
    }

    timer.stop();

    size_t time_filter = timer.getElapsed();

    count++;

    // FPS count
    fps_counter.stop();
    time_elapsed += (fps_counter.getElapsed()) / 1000000000.0;
    fps_counter.start();

    if (count % 10 == 0)
    {
      double fps = 10 / time_elapsed;
      time_elapsed = 0;
      cout << "FPS = " << fps << endl;
    }

    // Display results
    // if (gray.cols <= 1024 || gray.rows <= 1024)
    // {
    //   imshow("Input", gray);
    //   imshow("Blurred", out);
    //   if (count <= 1)
    //   {
    //     moveWindow("Blurred", WIDTH, 0);
    //   }
    //   key = waitKey(1);
    // }

    // Save results
    // if (gray.cols <= 1024 || gray.rows <= 1024)
    // {
    //   imwrite("Input_original.jpg", gray);
    //   imwrite("blur.jpg", out);
    // }
  }

  free(gray_ptr);
  free(out_ptr);
  cudaFree(gray_uni_ptr);
  cudaFree(out_uni_ptr);
  cudaFree(d_grayPtr);
  cudaFree(d_outPtr);

  return 0;
}
