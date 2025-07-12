/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


/* This code example is provided for illustration purposes for GTC 2025 talk:
   CUDA Techniques to Maximize Concurrency and System Utilization [S72686].
   APIs/code may change.
*/


/*
  To compile: nvcc -arch=sm_90 streams_and_green_contexts_example.cu -o streams_and_gc_example -lcuda
  Nsight Systems profile collected via: CUDA_VISIBLE_DEVICES=0 CUDA_MODULE_LOADING=EAGER nsys profile --cuda-event-trace=false -o  ./streams_and_gc_example
*/

#include <iostream>
#include <cuda.h>
#include <chrono>
#include <thread>


#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)


#define CU_CHECK(expr_to_check) do {            \
    CUresult result = expr_to_check;            \
    if(result != CUDA_SUCCESS)                  \
    {                                           \
        const char* pErrStr;                    \
        cuGetErrorString(result,&pErrStr);      \
        fprintf(stderr,                         \
                "CUDA Error: %s:%i:%s\n",       \
                __FILE__,                       \
                __LINE__,                       \
                pErrStr);                       \
    }                                           \
} while(0)


#define GPU_DEVICE_INDEX 0 // Update as needed
using namespace std::chrono_literals;

// Reference for %globaltimer: https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-globaltimer-globaltimer-lo-globaltimer-hi
__device__ __forceinline__ unsigned long long __globaltimer()
{
    unsigned long long globaltimer;
    // 64-bit global nanosecond timer
    asm volatile("mov.u64 %0, %globaltimer;"
                 : "=l"(globaltimer));
    return globaltimer;
}


__global__ void delay_kernel_us(uint32_t delay_us)
{
    // 64-bit global nanosecond timer
    constexpr uint64_t NS_PER_US = 1000UL;

    uint64_t start_time = __globaltimer();
    uint64_t end_time   = start_time + (delay_us * NS_PER_US);

    while(__globaltimer() < end_time)
    {
    };
}

__global__ void critical_kernel(uint32_t* __restrict__ d_input_buffer, uint32_t* __restrict__ d_output_buffer, int N, uint32_t multiplier)
{
    // Some critical work (for illustration purposes)
    for (int tid = threadIdx.x;  tid < N; tid += blockDim.x *gridDim.x) {
        d_output_buffer[tid] = d_input_buffer[tid] * multiplier;
    }

    // Also forcing some delay so it's easier to see kernel's duration in the timeline, compared to the longer running delay_kernel
    // 64-bit global nanosecond timer
    constexpr uint64_t NS_PER_US = 1000UL;
    constexpr uint64_t DELAY_IN_NS = 10*NS_PER_US; // can update delay as needed

    uint64_t start_time = __globaltimer();
    uint64_t end_time   = start_time + DELAY_IN_NS;

    // 64-bit timer has a long range so skipping wrap around check
    while(__globaltimer() < end_time)
    {
    };
}


struct useful_device_info {
   int SM_count; // number of SMs in GPU
   int max_threads_per_SM; // max. number of threads per SM
   int max_threads_per_block; // max. number of threads per thread block
   int cc_minor; //major compute capability version
   int cc_major; //minor compute capability version
   int max_dyn_shared_mem_opt_in;  // max. dynamic shared memory opt-in per thread block
   int max_shared_mem_per_SM; // max. shared memory per SM
};


void populate_useful_device_info(int gpu_device_index, useful_device_info& device_info) {
   device_info.SM_count = 0;
   device_info.max_threads_per_SM = 0;
   device_info.max_threads_per_block = 0;
   device_info.cc_minor = 0;
   device_info.cc_major = 0;
   device_info.max_dyn_shared_mem_opt_in = 0;
   CUDA_CHECK(cudaDeviceGetAttribute(&device_info.SM_count, cudaDevAttrMultiProcessorCount, gpu_device_index));
   CUDA_CHECK(cudaDeviceGetAttribute(&device_info.max_threads_per_SM, cudaDevAttrMaxThreadsPerMultiProcessor, gpu_device_index));
   CUDA_CHECK(cudaDeviceGetAttribute(&device_info.max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, gpu_device_index));
   CUDA_CHECK(cudaDeviceGetAttribute(&device_info.max_dyn_shared_mem_opt_in, cudaDevAttrMaxSharedMemoryPerBlockOptin, gpu_device_index));
   CUDA_CHECK(cudaDeviceGetAttribute(&device_info.cc_minor, cudaDevAttrComputeCapabilityMinor, gpu_device_index));
   CUDA_CHECK(cudaDeviceGetAttribute(&device_info.cc_major, cudaDevAttrComputeCapabilityMajor, gpu_device_index));
   CUDA_CHECK(cudaDeviceGetAttribute(&device_info.max_shared_mem_per_SM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, gpu_device_index));
   printf("GPU with %d.%d CC (compute capability) has %d total SMs, supports %d max. threads per SM, %d max threads per thread block and %d max dyn. shared memory size per block opt in and %d max shared memory per SM.\n", device_info.cc_major, device_info.cc_minor, device_info.SM_count, device_info.max_threads_per_SM, device_info.max_threads_per_block, device_info.max_dyn_shared_mem_opt_in, device_info.max_shared_mem_per_SM);
};

enum kernelRunCfgs {
    DELAY_KERNEL = 0,
    CRITICAL_KERNEL = 1,
    CRITICAL_KERNEL_HIGHER_STREAM_PRIO = 2,
    TOTAL_CFGS = 3
};

#define N 1024*1024*10 // array sizes relevant for critical kernel; adjust as needed
#define DELAY_IN_US 1000 // 1ms; adjust as needed
#define DELAY_TO_LAUNCH_CRITICAL_KERNEL_IN_MS 5 // 5ms
#define DELAY_KERNEL_WAVES_PER_SM_FOR_FULL_GPU 10

void launch_kernels(useful_device_info& device_info, cudaStream_t& strm_for_delay_kernel, cudaStream_t& strm_for_critical_kernel, cudaEvent_t timing_events[], uint32_t* d_input_buffer, uint32_t* d_output_buffer, bool print_timing=false) {

    size_t delay_kernel_dyn_shmem = device_info.max_dyn_shared_mem_opt_in; // to ensure only one thread block can run per SM
    int delay_kernel_grid_size    =  DELAY_KERNEL_WAVES_PER_SM_FOR_FULL_GPU * device_info.SM_count;
    int delay_kernel_block_size   = device_info.max_threads_per_block;

    size_t critical_kernel_dyn_shmem = 0;
    int critical_kernel_grid_size    = 256;
    int critical_kernel_block_size   = 256;

    int max_active_blocks_per_SM_for_delay_kernel = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_SM_for_delay_kernel, delay_kernel_us, delay_kernel_block_size, delay_kernel_dyn_shmem));

    int max_active_blocks_per_SM_for_critical_kernel = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_SM_for_critical_kernel, critical_kernel, critical_kernel_block_size, critical_kernel_dyn_shmem));
    if (!print_timing) {
        printf("max active blocks per SM for delay kernel is %d\n", max_active_blocks_per_SM_for_delay_kernel);
        printf("max active blocks per SM for critical kernel is %d\n", max_active_blocks_per_SM_for_critical_kernel);
    }

    // Launching each kernel once, so it's easier to see in an Nsight Systems trace.
    CUDA_CHECK(cudaEventRecord(timing_events[0], strm_for_delay_kernel));
    delay_kernel_us<<<dim3(delay_kernel_grid_size), dim3(delay_kernel_block_size), delay_kernel_dyn_shmem, strm_for_delay_kernel>>>(DELAY_IN_US);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(timing_events[1], strm_for_delay_kernel));

    std::this_thread::sleep_for(std::chrono::milliseconds(DELAY_TO_LAUNCH_CRITICAL_KERNEL_IN_MS)); // some delay, can update as needed

    CUDA_CHECK(cudaEventRecord(timing_events[2], strm_for_critical_kernel));
    critical_kernel<<<critical_kernel_grid_size, critical_kernel_block_size, critical_kernel_dyn_shmem, strm_for_critical_kernel>>>(d_input_buffer, d_output_buffer, N, 10 /* some multiplier*/);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(timing_events[3], strm_for_critical_kernel));

    CUDA_CHECK(cudaStreamSynchronize(strm_for_delay_kernel));
    CUDA_CHECK(cudaStreamSynchronize(strm_for_critical_kernel));

    if (print_timing) {
        float delay_kernel_duration_in_ms = 0.0f, critical_kernel_duration_in_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&delay_kernel_duration_in_ms, timing_events[0], timing_events[1]));
        CUDA_CHECK(cudaEventElapsedTime(&critical_kernel_duration_in_ms, timing_events[2], timing_events[3]));

        printf("   - delay_kernel duration %3.4f ms. Launched %d thread blocks (each with a %d usec duration) on a GPU with %d max SMs. This kernel can have at most %d active blocks per SM.\n",
            delay_kernel_duration_in_ms, delay_kernel_grid_size, DELAY_IN_US, device_info.SM_count, max_active_blocks_per_SM_for_delay_kernel);
        // The duration of the critical kernel is not just the actual GPU duration, but can also encompass time spent waiting for resources before it can start, etc.
        // Refer to Nsight Systems trace for a better view.
        printf("   - critical_kernel duration %3.4f ms. Launched %d thread blocks on a GPU with %d max SMs. This kernel can have at most %d active blocks per SM.\n",
            critical_kernel_duration_in_ms, critical_kernel_grid_size, device_info.SM_count, max_active_blocks_per_SM_for_critical_kernel);
    }
}


void launch_kernels_in_green_contexts(useful_device_info& device_info, cudaEvent_t timing_events[], uint32_t* d_input_buffer, uint32_t* d_output_buffer, bool print_timing=false) {

    CUdevice current_device;
    CU_CHECK(cuDeviceGet(&current_device, GPU_DEVICE_INDEX));

    // Get initial GPU resources, these are the SM (streaming multiprocessor) resources we will later split.
    CUdevResource initial_device_GPU_resources = {};
    CUdevResourceType default_resource_type = CU_DEV_RESOURCE_TYPE_SM;
    CU_CHECK(cuDeviceGetDevResource(current_device, &initial_device_GPU_resources, default_resource_type));
    printf("   Initial GPU resources retrieved via cuDeviceGetDevResource() have type %d and SM count %d.\n",  initial_device_GPU_resources.type, initial_device_GPU_resources.sm.smCount);

    // To make it generalizable across different archs, pick min SM count per group so there are 8 groups.
    int total_SMs = initial_device_GPU_resources.sm.smCount;
    const int min_SM_granularity = 2; // note that 2 would require the special use_flags for sm_90 (see below) or we'd get 8 SMs per group
    int min_requested_SMs_per_group = total_SMs / 8;
    min_requested_SMs_per_group -= (min_requested_SMs_per_group % min_SM_granularity);
    int requested_SMs[2] = {7*min_requested_SMs_per_group, min_requested_SMs_per_group};

    unsigned int use_flags = CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING; // or 0
    CUdevResource actual_split_result[8] = {}; // size of 8 needed here for this example; update as needed
    unsigned int requested_split_groups = (requested_SMs[0] + requested_SMs[1])/min_requested_SMs_per_group; // evenly divisible in this case
    unsigned int actual_split_groups = requested_split_groups;
    CU_CHECK(cuDevSmResourceSplitByCount(&actual_split_result[0], &actual_split_groups, &initial_device_GPU_resources, nullptr /* do not care about remaining group*/, use_flags, min_requested_SMs_per_group));
    printf("   Resources were split into %d resource groups (had requested %d) with %d SMs each (had requested %d)\n", actual_split_groups, requested_split_groups,
		    actual_split_result[0].sm.smCount, min_requested_SMs_per_group);

    CUdevResourceDesc resource_desc[2];
    int groups_for_GC[2];
    CUgreenCtx my_green_ctx[2];
    CUstream my_green_ctx_stream[2]; /* Reminder that CUstream and cudaStream_t can be used interchangeably. See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER.html#group__CUDART__DRIVER */
    int green_ctx_creation_flags = CU_GREEN_CTX_DEFAULT_STREAM;
    unsigned int green_ctx_stream_creation_flags = CU_STREAM_NON_BLOCKING;
    int priority = 0;
    cudaEvent_t my_green_ctx_start_stop_events[4]; // Reminder that "the types CUevent and cudaEvent_t are identical and may be used interchangeably."

    for (int i = 0; i < 2; i++) {
       groups_for_GC[i] = requested_SMs[i] / min_requested_SMs_per_group; // evenly divisible in this case
       int start_index = (i == 0) ? 0: groups_for_GC[0];
       // Generate descriptor by combining multiple resources
       CU_CHECK(cuDevResourceGenerateDesc(&resource_desc[i], &actual_split_result[start_index], groups_for_GC[i]));
       printf("   For the resource descriptor of green context %d, we will combine %d resources of %d SMs each\n", i+1, groups_for_GC[i], actual_split_result[start_index].sm.smCount); // all resources are uniform in size.
       // Create green context and stream for that green context
       CU_CHECK(cuGreenCtxCreate(&my_green_ctx[i], resource_desc[i], current_device, green_ctx_creation_flags));
       CU_CHECK(cuGreenCtxStreamCreate(&my_green_ctx_stream[i], my_green_ctx[i], green_ctx_stream_creation_flags, priority));
       CUDA_CHECK(cudaEventCreate(&my_green_ctx_start_stop_events[2*i]));
       CUDA_CHECK(cudaEventCreate(&my_green_ctx_start_stop_events[2*i+1]));
    }

    // -----------------------------------------------------------
    // Launching the work on green ctxs & streams created
    // -----------------------------------------------------------
    size_t delay_kernel_dyn_shmem = device_info.max_dyn_shared_mem_opt_in;
    int delay_kernel_grid_size    =  DELAY_KERNEL_WAVES_PER_SM_FOR_FULL_GPU *device_info.SM_count;
    int delay_kernel_block_size   = device_info.max_threads_per_block;

    size_t critical_kernel_dyn_shmem = 0;
    int critical_kernel_grid_size    = 256;
    int critical_kernel_block_size   = 256;

    int max_active_blocks_per_SM_for_delay_kernel = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_SM_for_delay_kernel, delay_kernel_us, delay_kernel_block_size, delay_kernel_dyn_shmem));

    int max_active_blocks_per_SM_for_critical_kernel = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_SM_for_critical_kernel, critical_kernel, critical_kernel_block_size, critical_kernel_dyn_shmem));

    CU_CHECK(cuGreenCtxRecordEvent(my_green_ctx[0], my_green_ctx_start_stop_events[0]));
    delay_kernel_us<<<dim3(delay_kernel_grid_size), dim3(delay_kernel_block_size), delay_kernel_dyn_shmem, my_green_ctx_stream[0]>>>(DELAY_IN_US);
    CUDA_CHECK(cudaGetLastError());
    CU_CHECK(cuGreenCtxRecordEvent(my_green_ctx[0], my_green_ctx_start_stop_events[1]));

    std::this_thread::sleep_for(std::chrono::milliseconds(DELAY_TO_LAUNCH_CRITICAL_KERNEL_IN_MS)); // some delay, can update as needed

    CU_CHECK(cuGreenCtxRecordEvent(my_green_ctx[1], my_green_ctx_start_stop_events[2]));
    critical_kernel<<<critical_kernel_grid_size, critical_kernel_block_size, critical_kernel_dyn_shmem, my_green_ctx_stream[1]>>>(d_input_buffer, d_output_buffer, N, 10);
    CUDA_CHECK(cudaGetLastError());
    CU_CHECK(cuGreenCtxRecordEvent(my_green_ctx[1], my_green_ctx_start_stop_events[3]));

    CUDA_CHECK(cudaEventSynchronize(my_green_ctx_start_stop_events[1]));
    CUDA_CHECK(cudaEventSynchronize(my_green_ctx_start_stop_events[3]));

    float delay_kernel_duration_in_ms = 0.0f, critical_kernel_duration_in_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&delay_kernel_duration_in_ms, my_green_ctx_start_stop_events[0], my_green_ctx_start_stop_events[1]));
    CUDA_CHECK(cudaEventElapsedTime(&critical_kernel_duration_in_ms, my_green_ctx_start_stop_events[2], my_green_ctx_start_stop_events[3]));

    CUdevResource resources_for_green_ctx[2] = {};
    CU_CHECK(cuGreenCtxGetDevResource(my_green_ctx[0], &resources_for_green_ctx[0], CU_DEV_RESOURCE_TYPE_SM));
    CU_CHECK(cuGreenCtxGetDevResource(my_green_ctx[1], &resources_for_green_ctx[1], CU_DEV_RESOURCE_TYPE_SM));
    printf("   - delay_kernel duration %3.4f ms. Launched %d thread blocks (each with a %d usec duration) on a GPU with %d max SMs but %d SMs for this GC. This kernel can have at most %d active blocks per SM.\n",
            delay_kernel_duration_in_ms, delay_kernel_grid_size, DELAY_IN_US, device_info.SM_count, resources_for_green_ctx[0].sm.smCount, max_active_blocks_per_SM_for_delay_kernel);

    printf("   - critical_kernel duration %3.4f ms. Launched %d thread blocks on a GPU with %d max SMs but %d SMs for this GC. This kernel can have at most %d active blocks per SM.\n",
            critical_kernel_duration_in_ms, critical_kernel_grid_size, device_info.SM_count, resources_for_green_ctx[1].sm.smCount, max_active_blocks_per_SM_for_critical_kernel);

    // Cleanup resources
    for (int i = 0; i < 4; i++) {
        CUDA_CHECK(cudaEventDestroy(my_green_ctx_start_stop_events[i]));
    }

    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaStreamDestroy(my_green_ctx_stream[i]));
        CU_CHECK(cuGreenCtxDestroy(my_green_ctx[i]));
    }
}



int main() {

    CUDA_CHECK(cudaSetDevice(GPU_DEVICE_INDEX));

    // Retrieve relevant device information
    useful_device_info device_info;
    populate_useful_device_info(GPU_DEVICE_INDEX, device_info);

    int block_size = device_info.max_threads_per_block;

    // These examples will launch a delay kernel with the max. possible dynamic shared memory per thread block (achieved via cudaFuncSetAttribute opt-in), so as to force
    // a single thread block per SM. This is so it can be easier to demonstrate, via timing measurements, the use of fewer SMs.
    CUDA_CHECK(cudaFuncSetAttribute(delay_kernel_us, cudaFuncAttributeMaxDynamicSharedMemorySize, device_info.max_dyn_shared_mem_opt_in)); // explicit opt-in needed for shared memory > 48KiB
    int max_active_thread_blocks_per_SM = 0;
    // thread block size used will not matter given max. dynamic shared memory will force single thread block per SM; any value till useful_device_info.max_threads_per_block would work
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_thread_blocks_per_SM, delay_kernel_us, block_size, device_info.max_dyn_shared_mem_opt_in));
    if (max_active_thread_blocks_per_SM != 1) {
	    printf("Delay kernel: Max active thread blocks per SM %d, but expected 1! Exiting.\n", max_active_thread_blocks_per_SM);
	    return 1;
    }

    // Get stream priorities ranges
    int lowest_stream_prio = 0;
    int greatest_stream_prio = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowest_stream_prio, &greatest_stream_prio));
    //printf("Lowest stream priority %d while highest is %d\n", lowest_stream_prio, greatest_stream_prio);

    // Create non blocking streams for all cases but green contexts
    cudaStream_t stream[kernelRunCfgs::TOTAL_CFGS];
    for (int i = 0; i < kernelRunCfgs::TOTAL_CFGS; i++) {
	if (i == kernelRunCfgs::CRITICAL_KERNEL_HIGHER_STREAM_PRIO) {
            CUDA_CHECK(cudaStreamCreateWithPriority(&stream[i], cudaStreamNonBlocking, greatest_stream_prio));
	} else {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking)); // lowest_stream_prio of 0 will be used
	}
    }

    cudaEvent_t timing_events[4]; // start-stop for 2 kernels;
    for (int i = 0; i < 4; i++) {
        CUDA_CHECK(cudaEventCreate(&timing_events[i]));
    }

    // Memory allocations, initialization.
    uint32_t *h_input_buffer,  *d_input_buffer, *d_output_buffer;
    CUDA_CHECK(cudaMallocHost(&h_input_buffer, sizeof(uint32_t)* N));
    CUDA_CHECK(cudaMalloc(&d_input_buffer, sizeof(uint32_t)* N));
    CUDA_CHECK(cudaMalloc(&d_output_buffer, sizeof(uint32_t)* N));
    for (int i = 0; i < N; i++) h_input_buffer[i] = i; // init to some value
    CUDA_CHECK(cudaMemcpyAsync(d_input_buffer, h_input_buffer, sizeof(uint32_t) * N, cudaMemcpyHostToDevice, stream[kernelRunCfgs::CRITICAL_KERNEL]));
    CUDA_CHECK(cudaStreamSynchronize(stream[kernelRunCfgs::CRITICAL_KERNEL]));

    // One could potentially add NVTX ranges to see each experiment in the nsys profile more clearly

    // Run each kernel once; ignore timings
    printf("Run each kernel once; no timings printed\n");
    launch_kernels(device_info, stream[kernelRunCfgs::DELAY_KERNEL], stream[kernelRunCfgs::CRITICAL_KERNEL], timing_events, d_input_buffer, d_output_buffer, false);

    // --------------------------------------------------------------------------------------------------------
    // Run each kernel in isolation (serializing them on the same CUDA stream); nothing else running on the GPU
    // --------------------------------------------------------------------------------------------------------

    printf("\nExample 1: Run each kernel in isolation by serializing them on the same CUDA stream; nothing else running on the GPU\n");
    launch_kernels(device_info, stream[kernelRunCfgs::CRITICAL_KERNEL], stream[kernelRunCfgs::CRITICAL_KERNEL], timing_events, d_input_buffer, d_output_buffer, true);

    // -----------------------------------------------------------
    // Now take the measurement when both kernels are running at the same time; critical kernel is triggered 5ms after the start of the first one
    // Highlight how since there's no GPU work on strm_for_critical_kernel, the elapsed time measured will also include the time waiting for SM
    // resources to be available.
    // Also depending on whether critical_kernel is launched on a stream with higher priority than the delay_kernel or not,
    // it may start executing as soon as some thread blocks from delay_kernel complete or until all CTAs from delay_kernel complete.
    // -----------------------------------------------------------
    printf("\nExample 2: Launch kernels on separate non blocking streams with the same priority. delay_kernel launched first; critical kernel launched %d ms later\n", DELAY_TO_LAUNCH_CRITICAL_KERNEL_IN_MS);
    launch_kernels(device_info, stream[kernelRunCfgs::DELAY_KERNEL], stream[kernelRunCfgs::CRITICAL_KERNEL], timing_events, d_input_buffer, d_output_buffer, true);

    // Repeat previous experiment but now have the critical kernel run on the high priority stream
    printf("\nExample 3: Launch kernels on separate non blocking streams similar to example 2, but launch critical_kernel on a stream with higher priority\n");
    launch_kernels(device_info, stream[kernelRunCfgs::DELAY_KERNEL], stream[kernelRunCfgs::CRITICAL_KERNEL_HIGHER_STREAM_PRIO], timing_events, d_input_buffer, d_output_buffer, true);


    // -----------------------------------------------------------
    // Now create two non overlapping green contexts and streams in them and launch the kernels.
    // No green context can use all the SMs of the GPU, so the duration of the long running delay_kernel will be affected, but
    // now there will always be some SMs available for the the critical_kernel to start running immediately
    // -----------------------------------------------------------
    printf("\nExample 4: Launch kernels on separate non blocking streams of the same priority belonging to different green contexts\n");
    launch_kernels_in_green_contexts(device_info, timing_events, d_input_buffer, d_output_buffer, true);


    // Cleanup resources
    for (int i = 0; i < 4; i++) {
        CUDA_CHECK(cudaEventDestroy(timing_events[i]));
    }

    CUDA_CHECK(cudaFreeHost(h_input_buffer));
    CUDA_CHECK(cudaFree(d_input_buffer));
    CUDA_CHECK(cudaFree(d_output_buffer));

    for (int i = 0; i < kernelRunCfgs::TOTAL_CFGS; i++) {
        CUDA_CHECK(cudaStreamDestroy(stream[i]));
    }

    return 0;
}

