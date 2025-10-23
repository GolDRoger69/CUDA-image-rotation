#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

#include <iostream>
#include <filesystem>
#include <cuda_runtime.h>

using namespace std;
namespace fs = std::filesystem;

__global__ void gaussian_blur_kernel(unsigned char*, unsigned char*, int, int);

void process_image(const string& input_path, const string& output_path) {
    int width, height, channels;
    unsigned char* h_input = stbi_load(input_path.c_str(), &width, &height, &channels, 1);
    if (!h_input) {
        cerr << "Failed to load: " << input_path << endl;
        return;
    }

    size_t img_size = width * height * sizeof(unsigned char);
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);
    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    gaussian_blur_kernel<<<blocks, threads>>>(d_input, d_output, width, height);

    unsigned char* h_output = new unsigned char[width * height];
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);
    stbi_write_png(output_path.c_str(), width, height, 1, h_output, width);

    cudaFree(d_input); cudaFree(d_output);
    stbi_image_free(h_input); delete[] h_output;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: ./image_processor <input_dir> <output_dir>\n";
        return 1;
    }

    string input_dir = argv[1];
    string output_dir = argv[2];

    for (const auto& file : fs::directory_iterator(input_dir)) {
        if (file.path().extension() == ".png") {
            string input_path = file.path().string();
            string output_path = output_dir + "/" + file.path().filename().string();
            cout << "Processing " << file.path().filename().string() << endl;
            process_image(input_path, output_path);
        }
    }

    return 0;
}
