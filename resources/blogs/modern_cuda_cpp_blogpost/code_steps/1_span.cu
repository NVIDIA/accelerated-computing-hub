#include <cuda_runtime.h>
#include <omp.h>

#include <iostream>
#include <vector>
#include <algorithm>

#include <cuda/std/mdspan>
#include <cuda/cmath>
#include <cuda/stream>
#include <cuda/launch>
#include <cuda/algorithm>
#include <cuda/std/span>
#include <cuda/mdspan>

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

// Alias for a 2 dimensions mdspan
template <typename T>
using span_2d = cuda::std::mdspan<T, cuda::std::dims<2>>;

// Kernel converting the red, green and blue images into a single gray image
template <typename Configuration>
__global__ void computeRGBToGray(Configuration config, span_2d<const pixel_t> d_image_r, span_2d<const pixel_t> d_image_g, span_2d<const pixel_t> d_image_b, span_2d<pixel_t> d_image_gray) {
    // Compute the thread global index in the grid using the configuration
    const auto [x, y, _] = cuda::gpu_thread.index(cuda::grid, config);

    // Boundary check selecting only threads within the image boundary
    if (y < d_image_gray.extent(0) && x < d_image_gray.extent(1)) {
        // Convert from rgb to grayscale and store the result in global memory
        d_image_gray(y, x) = static_cast<pixel_t>(0.299f * d_image_r(y, x) + 0.587f * d_image_g(y, x) + 0.114f * d_image_b(y, x));
    }
}

// Kernel computing the median of each tile in the grayscale image
template <int TILE_WIDTH, int HISTO_SIZE, typename Configuration>
__global__ void computeMedian(Configuration config, span_2d<const pixel_t> d_image_gray, span_2d<pixel_t> d_median) {
    // Compute the thread global index in the grid using the configuration
    const auto [x, y, _] = cuda::gpu_thread.index(cuda::grid, config);

    // Boundary check selecting only threads within the image boundary
    if (!(y < d_image_gray.extent(0) && x < d_image_gray.extent(1)))
        return;

    // Allocate the shared memory in which we will store the tile and view it as a 2D mdspan
    __shared__ pixel_t shared[TILE_WIDTH * TILE_WIDTH];
    // We can view the shared memory as a non owning 2D mdspan
    // In debug mode this protects against out of bounds access
    cuda::shared_memory_mdspan tile_2d(shared, TILE_WIDTH, TILE_WIDTH);

    // Compute the thread index within the thread block to address shared memory
    const auto block_idx = cuda::gpu_thread.index(cuda::block, config);

    // Load the tile's grayscale value from global memory into shared memory
    tile_2d(block_idx.y, block_idx.x) = d_image_gray(y, x);

    // Synchronize to make sure all threads have loaded their data
    __syncthreads();

    // Sort the tile array using a single threaded bubble sort
    // While sorting its easier to see the tile as a 1D array
    cuda::shared_memory_mdspan tile_1d(shared, TILE_WIDTH * TILE_WIDTH);
    if (block_idx.x == 0 && block_idx.y == 0) {
        for (int i = 0; i < TILE_WIDTH * TILE_WIDTH; ++i)
            for (int j = i + 1; j < TILE_WIDTH * TILE_WIDTH; ++j)
                if (tile_1d(i) > tile_1d(j))
                    cuda::std::swap(tile_1d(i), tile_1d(j));

        // Each thread block stores the median, found in the middle index after sorting, in the global median array
        const int medianIndex = (TILE_WIDTH * TILE_WIDTH) / 2;
        const auto grid_block_idx = cuda::block.index(cuda::grid, config);
        d_median(grid_block_idx.y, grid_block_idx.x) = tile_1d(medianIndex);
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

    // Explained at a later stage, unimportant for now
    cudaStream_t native;
    CUDA_CHECK_ERROR(cudaStreamCreate(&native));
    cuda::stream stream = cuda::stream::from_native_handle(native);    


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

        // Create the kernel launch configuration with static block dimensions
        auto config = cuda::make_config(
            cuda::block_dims<TILE_WIDTH, TILE_WIDTH>(),
            cuda::grid_dims(dim3(cuda::ceil_div(IMAGE_LENGTH, TILE_WIDTH), cuda::ceil_div(IMAGE_LENGTH, TILE_WIDTH))));

        // Launch a GPU kernel to convert the RGB images to grayscale
        cuda::launch(stream, config, computeRGBToGray<decltype(config)>,
            span_2d<const pixel_t>{d_image_r, IMAGE_LENGTH, IMAGE_LENGTH},
            span_2d<const pixel_t>{d_image_g, IMAGE_LENGTH, IMAGE_LENGTH},
            span_2d<const pixel_t>{d_image_b, IMAGE_LENGTH, IMAGE_LENGTH},
            span_2d<pixel_t>{d_image_gray, IMAGE_LENGTH, IMAGE_LENGTH});

        // Launch the GPU kernel to compute the median of every tile in the image
        cuda::launch(stream, config, computeMedian<TILE_WIDTH, HISTO_SIZE, decltype(config)>,
            span_2d<const pixel_t>{d_image_gray, IMAGE_LENGTH, IMAGE_LENGTH},
            span_2d<pixel_t>{d_median, NB_TILE_Y, NB_TILE_X});

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
