Convolutional filters are important tools in the field of Image and signal processing. These filters are usually highly compute intensive with algorithms often have $O(n^4)$ complexity. In this project we aim to accelerate several convolutional filters using various parallelization techniques on an NVIDIA GPU.

How to run?
    Source code and build tools have been arranged in the following folder hierarchy:
    ![image](https://github.com/suraj-2306/260CCUDA/assets/27968098/5a3feb42-a2b6-4795-a207-575480c4c0ea)
    
    Each folder has CMakeLists which can be used to generate make files to compile the source code. Modify the this file according to he machine where programs will be executed.

    Modify main.cpp within each folder to use the image/video of choice. Tests have been run using the ".raw" format.
        "VideoCapture cap(<input file>)"
    
    After compilations, use the following command to run each filter:

    1D Convolutional filter:
        ./1DFilter

    2D Blur filter:
        ./2DFilter <y-dimension> <x-dimension> <target>
        <target>: 0: CPU, 1: GPU, 2: GPU with unified memory
        e.g. ./2DFilter 768 768 1

    2D Sobel filter:
        ./SobelFilter <y-dimension> <x-dimension> <target>
        <target>: 0: OpenCV, 1: CPU, 2: GPU, 3: GPU with unified memory
        e.g. ./SobelFilter 768 768 2
