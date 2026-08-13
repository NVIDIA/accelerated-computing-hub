#include <cuda_runtime.h>
#include <omp.h>

#include <iostream>
#include <vector>
#include <algorithm>

#include <cuda/cmath>

#define CUDA_CHECK_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// Alias for an image pixel
using pixel_t = uint8_t;

// Kernel converting the red, green and blue images into a single gray image
__global__ void computeRGBToGray(const pixel_t* d_image_r, const pixel_t* d_image_g, const pixel_t* d_image_b, pixel_t* d_image_gray, int width, int height) {
    // Compute the thread global index in the grid
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Boundary check selecting only threads within the image boundary
    if (x < width && y < height) {
        // Compute the thread index in the image
        const int i = x + y * width;
        // Convert from rgb to grayscale and store the result in global memory
        d_image_gray[i] = static_cast<pixel_t>(0.299f * d_image_r[i] + 0.587f * d_image_g[i] + 0.114f * d_image_b[i]);
    }
}

// Kernel computing the median of each tile in the grayscale image
template <int TILE_WIDTH, int HISTO_SIZE>
__global__ void computeMedian(pixel_t *d_image_gray, pixel_t *d_median, int width, int height) {
    // Compute the thread global index in the grid
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Boundary check selecting only threads within the image boundary
    if (!(x < width && y < height))
        return;

    // Allocate the shared memory in which we will store the tile
    __shared__ pixel_t tile[TILE_WIDTH * TILE_WIDTH];

    // Compute the thread index in the image
    const int index = x + y * width;

    // Load the tile's grayscale value from global memory into shared memory
    tile[index] = d_image_gray[index];

    // Synchronize to make sure all threads have loaded their data
    __syncthreads();

    // Sort the tile array using a single threaded bubble sort
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < TILE_WIDTH * TILE_WIDTH; ++i)
            for (int j = i + 1; j < TILE_WIDTH * TILE_WIDTH; ++j)
                if (tile[i] > tile[j])
                    cuda::std::swap(tile[i], tile[j]);

        // Each thread block stores the median, found in the middle index after sorting, in the global median array
        const int medianIndex = (TILE_WIDTH * TILE_WIDTH) / 2;
        d_median[blockIdx.x + blockIdx.y * gridDim.x] = tile[medianIndex];
    }
}

int main() {
    // Define all the example constants
    constexpr auto TILE_WIDTH = 32;
    constexpr auto HISTO_SIZE = 256;
    constexpr auto NB_TILE_X = 250;
    constexpr auto NB_TILE_Y = NB_TILE_X;
    constexpr auto IMAGE_LENGTH = TILE_WIDTH * NB_TILE_X;
    constexpr auto IMAGE_SIZE = IMAGE_LENGTH * IMAGE_LENGTH;
    constexpr auto NB_IMAGES = 3;
    constexpr auto INIT_VALUE = 4;

    // Allocate the CPU memory to store the images tiles medians and for the red, green, blue and grayscale images
    std::vector<std::vector<pixel_t>> h_images_r(NB_IMAGES, std::vector<pixel_t>(IMAGE_SIZE, 4));
    std::vector<std::vector<pixel_t>> h_images_g(NB_IMAGES, std::vector<pixel_t>(IMAGE_SIZE, 4));
    std::vector<std::vector<pixel_t>> h_images_b(NB_IMAGES, std::vector<pixel_t>(IMAGE_SIZE, 4));
    std::vector<std::vector<pixel_t>> h_images_gray(NB_IMAGES, std::vector<pixel_t>(IMAGE_SIZE, 0));
    std::vector<std::vector<pixel_t>> h_medians(NB_IMAGES, std::vector<pixel_t>(NB_TILE_X * NB_TILE_Y));

    // Run the image processing pipeline for each image, in parallel
    #pragma omp parallel for
    for (int i = 0; i < NB_IMAGES; ++i)
    {
        pixel_t *d_image_r, *d_image_g, *d_image_b, *d_image_gray, *d_median;

        // Allocate the GPU memory for each container
        CUDA_CHECK_ERROR(cudaMalloc(&d_image_r, IMAGE_SIZE * sizeof(pixel_t)));
        CUDA_CHECK_ERROR(cudaMalloc(&d_image_g, IMAGE_SIZE * sizeof(pixel_t)));
        CUDA_CHECK_ERROR(cudaMalloc(&d_image_b, IMAGE_SIZE * sizeof(pixel_t)));
        CUDA_CHECK_ERROR(cudaMalloc(&d_image_gray, IMAGE_SIZE * sizeof(pixel_t)));
        CUDA_CHECK_ERROR(cudaMalloc(&d_median, (NB_TILE_X * NB_TILE_Y) * sizeof(pixel_t)));

        // Copy the memory of each container from CPU to GPU
        CUDA_CHECK_ERROR(cudaMemcpy(d_image_r, h_images_r[i].data(), IMAGE_SIZE * sizeof(pixel_t), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(d_image_g, h_images_g[i].data(), IMAGE_SIZE * sizeof(pixel_t), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(d_image_b, h_images_b[i].data(), IMAGE_SIZE * sizeof(pixel_t), cudaMemcpyHostToDevice));

        // Launch a GPU kernel to convert the RGB images to grayscale
        dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
        dim3 gridSize(cuda::ceil_div(IMAGE_LENGTH, blockSize.x), cuda::ceil_div(IMAGE_LENGTH, blockSize.y));
        computeRGBToGray<<<gridSize, blockSize>>>(d_image_r, d_image_g, d_image_b, d_image_gray, IMAGE_LENGTH, IMAGE_LENGTH);
        CUDA_CHECK_ERROR(cudaGetLastError());

        // Launch the GPU kernel to compute the median of every tile in the image
        computeMedian<TILE_WIDTH, HISTO_SIZE><<<gridSize, blockSize>>>(d_image_gray, d_median, IMAGE_LENGTH, IMAGE_LENGTH);
        CUDA_CHECK_ERROR(cudaGetLastError());

        // Copy the GPU median memory back to the CPU
        CUDA_CHECK_ERROR(cudaMemcpy(h_medians[i].data(), d_median, (NB_TILE_X * NB_TILE_Y) * sizeof(pixel_t), cudaMemcpyDeviceToHost));

        // Free the GPU memory
        CUDA_CHECK_ERROR(cudaFree(d_image_r));
        CUDA_CHECK_ERROR(cudaFree(d_image_g));
        CUDA_CHECK_ERROR(cudaFree(d_image_b));
        CUDA_CHECK_ERROR(cudaFree(d_image_gray));
        CUDA_CHECK_ERROR(cudaFree(d_median));
    }

    // Check the result for each image
    for (int image = 0; image < NB_IMAGES; ++image)
    {
        if (!std::all_of(h_medians[image].cbegin(), h_medians[image].cend(), [INIT_VALUE](pixel_t i){ return i == INIT_VALUE; }))
        {
            std::cout << "Value should be " << INIT_VALUE << std::endl;
            for (auto e : h_medians[image])
                std::cout << e << " ";
            std::cout << std::endl;
            return -1;
        }
    }

    std::cout << "All good" << std::endl;

    return 0;
}