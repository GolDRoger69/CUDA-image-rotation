// src/image_blur.cu
#include <cuda_runtime.h>

__global__ void gaussian_blur_kernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int kernel[3][3] = {{1,2,1}, {2,4,2}, {1,2,1}};
    int sum = 0;

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        int value = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                value += input[(y+dy)*width + (x+dx)] * kernel[dy+1][dx+1];
                sum += kernel[dy+1][dx+1];
            }
        }
        output[y*width + x] = value / sum;
    }
}
