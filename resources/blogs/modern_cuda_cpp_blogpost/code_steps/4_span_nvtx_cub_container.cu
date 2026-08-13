#include <cuda_runtime.h>
#include <omp.h>

#include <iostream>
#include <vector>
#include <algorithm>

#include <cuda/std/mdspan>
#include <cuda/cmath>

#include <nvtx3/nvtx3.hpp>

#include <cub/cub.cuh>

#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/stream>
#include <cuda/memory>
#include <cuda/launch>
#include <cuda/algorithm>
#include <cuda/std/span>

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

// Kernel computing the median of each tile in the grayscale image
template <int TILE_WIDTH, int HISTO_SIZE, typename Configuration>
__global__ void computeMedian(Configuration config, span_2d<const pixel_t> d_image_gray, span_2d<pixel_t> d_median) {
    // Compute the thread global index in the grid using the configuration
    const auto [x, y, _] = cuda::gpu_thread.index(cuda::grid, config);

    // Boundary check selecting only threads within the image boundary
    if (!(y < d_image_gray.extent(0) && x < d_image_gray.extent(1)))
        return;

    // Declare and allocate the storage for CUB BlockRadixSort
    using BlockRadixSort = cub::BlockRadixSort<pixel_t, TILE_WIDTH, 1, cub::NullType, 4, true, cub::BLOCK_SCAN_WARP_SCANS, cudaSharedMemBankSizeFourByte, TILE_WIDTH>;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    // Load the tile's grayscale value from global memory
    pixel_t thread_keys[1];
    thread_keys[0] = d_image_gray(y, x);

    // Perform the thread-block-level radix sort
    BlockRadixSort(temp_storage).Sort(thread_keys);

    // Select the thread found at the middle index
    // Write its value which is, after sorting, the median, in the global median array
    const auto block_idx = cuda::gpu_thread.index(cuda::block, config);
    if (block_idx.x == TILE_WIDTH / 2 && block_idx.y == TILE_WIDTH / 2) {
        const auto grid_block_idx = cuda::block.index(cuda::grid, config);
        d_median(grid_block_idx.y, grid_block_idx.x) = thread_keys[0];
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

    // Resource to handle the GPU memory allocations
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});

    nvtxRangePushA("Images compute");

    // Run the image processing pipeline for each image, in parallel
    #pragma omp parallel for
    for (int i = 0; i < NB_IMAGES; ++i)
    {
        // NVTX range tied to the for-loop scope to tag the whole image processing step
        nvtx3::scoped_range fun_scope("Image compute");

        // Pushing then popping an NVTX range for every action to tag them
        nvtxRangePushA("Memory Allocation");

        // Allocate the GPU memory using containers
        cuda::device_buffer<pixel_t> d_image_r = cuda::make_buffer<pixel_t>(stream, device_resource, IMAGE_SIZE, cuda::no_init);
        cuda::device_buffer<pixel_t> d_image_g = cuda::make_buffer<pixel_t>(stream, device_resource, IMAGE_SIZE, cuda::no_init);
        cuda::device_buffer<pixel_t> d_image_b = cuda::make_buffer<pixel_t>(stream, device_resource, IMAGE_SIZE, cuda::no_init);
        cuda::device_buffer<pixel_t> d_image_gray = cuda::make_buffer<pixel_t>(stream, device_resource, IMAGE_SIZE, cuda::no_init);
        cuda::device_buffer<pixel_t> d_median = cuda::make_buffer<pixel_t>(stream, device_resource, NB_TILE_X * NB_TILE_Y, cuda::no_init);

        nvtxRangePop();

        nvtxRangePushA("Memory Copy In");

        // Copy the memory of each container from CPU to GPU
        CUDA_CHECK_ERROR(cudaMemcpy(d_image_r.data(), h_images_r[i].data(), IMAGE_SIZE * sizeof(pixel_t), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(d_image_g.data(), h_images_g[i].data(), IMAGE_SIZE * sizeof(pixel_t), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(d_image_b.data(), h_images_b[i].data(), IMAGE_SIZE * sizeof(pixel_t), cudaMemcpyHostToDevice));

        nvtxRangePop();

        nvtxRangePushA("Kernel RGB to gray scale");

        // Use CUB to convert the RGB images to grayscale
        cub::DeviceTransform::Transform(cuda::std::make_tuple(d_image_r.cbegin(), d_image_g.cbegin(), d_image_b.cbegin()),
                                        d_image_gray.begin(),
                                        d_image_gray.size(),
                                        [] __host__ __device__ (pixel_t r, pixel_t g, pixel_t b) {
                                            return static_cast<pixel_t>(0.299f * r + 0.587f * g + 0.114f * b);
                                        },
                                        stream.get());

        nvtxRangePop();

        nvtxRangePushA("Kernel median");

        // Create the kernel launch configuration with static block dimensions
        auto config = cuda::make_config(
            cuda::block_dims<TILE_WIDTH, TILE_WIDTH>(),
            cuda::grid_dims(dim3(cuda::ceil_div(IMAGE_LENGTH, TILE_WIDTH), cuda::ceil_div(IMAGE_LENGTH, TILE_WIDTH))));

        // Launch the GPU kernel to compute the median of every tile in the image
        cuda::launch(stream, config, computeMedian<TILE_WIDTH, HISTO_SIZE, decltype(config)>,
            span_2d<const pixel_t>{d_image_gray.data(), IMAGE_LENGTH, IMAGE_LENGTH},
            span_2d<pixel_t>{d_median.data(), NB_TILE_Y, NB_TILE_X});

        nvtxRangePop();

        nvtxRangePushA("Memory Copy Out");

        // Copy the GPU median memory back to the CPU
        CUDA_CHECK_ERROR(cudaMemcpy(h_medians[i].data(), d_median.data(), (NB_TILE_X * NB_TILE_Y) * sizeof(pixel_t), cudaMemcpyDeviceToHost));

        nvtxRangePop();

        nvtxRangePop();
    }

    nvtxRangePop();

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
