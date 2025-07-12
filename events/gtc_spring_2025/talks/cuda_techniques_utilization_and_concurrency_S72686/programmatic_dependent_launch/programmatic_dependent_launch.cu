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
  To compile: nvcc -arch=sm_90 programmatic_dependent_launch.cu -o pdl_example
  Nsight Systems profile collected via: CUDA_VISIBLE_DEVICES=0 CUDA_MODULE_LOADING=EAGER nsys profile --cuda-event-trace=false -o  ./pdl_example
*/


/* Programmatic Dependent Launch (PDL) example.

   Overview:

   Three kernels launched on the same CUDA stream: primary_kernel -> secondary_kernel-> another_kernel

   Disclaimer: these kernels call a delay function and their code does not have an actual data dependence;
   this is just to more easily illustrate the overlap effect.

   primary_kernel: delay ~100us; trigger via PDL launch of secondary; delay 50us
                   CTA (thread block) count is such that there are 2 waves for the full GPU

   secondary kernel: ~100us delay followed by grid sync (PDL), followed by some computations and PDL trigger of another kernel /grid sync/ some computations/programmatic launch of another_kernel/50us delay
                     kernel should be able to coexist with primary

    another_kernel:  20us delay/grid dep sync/50us delay


    Expectations:
    - Assuming resources available, secondary kernel could start ~50us before the end of the last wave of the primary kernel
    - Assuming resources available, another_kernel could start ~50us before the end of the secondary kernel
    - another_kernel's prologue, before grid sync, has a shorter duration than the work done by the secondary kernel after trigger; so another_kernel's duration will increase compared
to its duration without PDL
*/

#include <iostream>


#define ITER_CNT 10


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
    exit(1);                                  \
    }\
} while(0)


#define GPU_DEVICE_INDEX 0 // update as needed
#define N 1000000 // does not matter, as limited by thread blocks


// Reference for %globaltimer: https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-globaltimer-globaltimer-lo-globaltimer-hi
__device__ __forceinline__ unsigned long long __globaltimer()
{
    unsigned long long globaltimer;
    // 64-bit global nanosecond timer
    asm volatile("mov.u64 %0, %globaltimer;"
                 : "=l"(globaltimer));
    return globaltimer;
}


__device__ __forceinline__ void delay_in_us(uint32_t delay_us)
{
    // 64-bit global nanosecond timer
    constexpr uint64_t NS_PER_US = 1000UL;

    uint64_t start_time = __globaltimer();
    uint64_t end_time   = start_time + (delay_us * NS_PER_US);

    while(__globaltimer() < end_time)
    {
    };
}


__global__ void primary_kernel() {
   delay_in_us(100);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
   //if (threadIdx.x == 0) // one thread per block suffices
   cudaTriggerProgrammaticLaunchCompletion();
#endif

   delay_in_us(50);
}

__global__ void primary_kernel_all_blocks_skip_trigger() {
   delay_in_us(100);
   if (false) {
       // If no block calls trigger, then the secondary kernel will get triggered only when all warps of this kernel complete their execution
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
   //if (threadIdx.x == 0) // one thread per block suffices
   cudaTriggerProgrammaticLaunchCompletion();
#endif
   }

   delay_in_us(50);
}


__global__ void secondary_kernel(uint8_t* d_buffer) {
   delay_in_us(100);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
   cudaGridDependencySynchronize();
#endif

   // Just some computations
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   if (tid < N) {
       d_buffer[tid] = 10;
   }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
   cudaTriggerProgrammaticLaunchCompletion();
#endif
   delay_in_us(50);
}


__global__ void another_kernel() {
   delay_in_us(20);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
   cudaGridDependencySynchronize();
#endif
   delay_in_us(50);
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


void pdl_examples(int num_iters) {

    CUDA_CHECK(cudaSetDevice(GPU_DEVICE_INDEX));
    cudaStream_t strm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
    uint8_t* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(uint8_t)* N));

    // Retrieve relevant device information
    useful_device_info device_info;
    populate_useful_device_info(GPU_DEVICE_INDEX, device_info);

    if (device_info.cc_major < 9) {
       printf("Programmatic Dependent Launch (PDL) requires compute capability >= 9.0. Your GPU has CC %d.%d. Exiting early.\n", device_info.cc_major, device_info.cc_minor);
       exit(1);
    }

    int max_active_thread_blocks_per_SM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_thread_blocks_per_SM,  primary_kernel, device_info.max_threads_per_block, 0));
    printf("Primary kernel: Max active thread blocks per SM %d\n", max_active_thread_blocks_per_SM);

    int primary_kernel_blocks = 2 * max_active_thread_blocks_per_SM * device_info.SM_count; // 2 waves per GPU
    int primary_kernel_threads = device_info.max_threads_per_block;

    max_active_thread_blocks_per_SM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_thread_blocks_per_SM,  primary_kernel_all_blocks_skip_trigger, device_info.max_threads_per_block, 0));
    printf("Primary kernel variant where all blocks skip calling trigger: Max active thread blocks per SM %d\n", max_active_thread_blocks_per_SM);

    int secondary_kernel_blocks =  2 * device_info.SM_count; // don't need all SMs to illustrate behavior
    int secondary_kernel_threads = 32;

    max_active_thread_blocks_per_SM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_thread_blocks_per_SM,  secondary_kernel, secondary_kernel_threads, 0));
    printf("secondary kernel and another_kernel can have %d max active CTAs per SM\n", max_active_thread_blocks_per_SM);

    // Set up launch config of secondary kernel
    cudaLaunchConfig_t  config_secondary;
    config_secondary.blockDim = dim3(secondary_kernel_threads);
    config_secondary.gridDim = dim3(secondary_kernel_blocks);
    config_secondary.dynamicSmemBytes = 0;
    config_secondary.stream = strm; // Putting a different stream here won't cause an error, but it won't be PDL
    config_secondary.numAttrs = 1;

    cudaLaunchAttribute attrs_secondary[1];
    attrs_secondary[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs_secondary[0].val.programmaticStreamSerializationAllowed = 1; // setting this to 0, will show no overlap
    config_secondary.attrs = &attrs_secondary[0];

    // Set up launch config of third kernel
    cudaLaunchConfig_t  config_last;
    config_last.blockDim = dim3(secondary_kernel_threads);
    config_last.gridDim = dim3(secondary_kernel_blocks);
    config_last.dynamicSmemBytes = 0;
    config_last.stream = strm; // Putting a different stream here won't cause an error, but it won't be PDL
    config_last.numAttrs = 1;

    cudaLaunchAttribute attrs_last[1];
    attrs_last[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs_last[0].val.programmaticStreamSerializationAllowed = 1; // setting this to 0, will show no overlap
    config_last.attrs = &attrs_last[0];

    // First, show example use without PDL
    printf("\nExample 1: Launch 3 kernels on the same CUDA stream without PDL\n");
    for (int i = 0; i < num_iters; i++) { // run back to back iterations

        primary_kernel<<<primary_kernel_blocks, primary_kernel_threads,  0, strm>>>();
        CUDA_CHECK(cudaGetLastError());

        secondary_kernel<<<secondary_kernel_blocks, secondary_kernel_threads,  0, strm>>>(d_ptr);
        CUDA_CHECK(cudaGetLastError());

        another_kernel<<<secondary_kernel_blocks, secondary_kernel_threads,  0, strm>>>();
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(strm)); // not strictly needed

    // Example to illustrate correct PDL use.
    printf("\nExample 2: Use PDL\n");
    for (int i = 0; i < num_iters; i++) { // run back to back iterations

       // Launch primary kernel followed by secondary_kernel and then another_kernel
       primary_kernel<<<primary_kernel_blocks, primary_kernel_threads,  0, strm>>>();
       CUDA_CHECK(cudaGetLastError());

       CUDA_CHECK(cudaLaunchKernelEx(&config_secondary, secondary_kernel, d_ptr));
       CUDA_CHECK(cudaLaunchKernelEx(&config_last, another_kernel));
    }
    CUDA_CHECK(cudaStreamSynchronize(strm)); // not strictly needed


    // This example illustrates the behavior if the secondary kernel is implicitly triggered once all primary kernel's warps complete, because
    // not all CTAs called PDL.
    printf("\nExample 3: Use PDL, but have a different primary kernel variant where all CTAs skip calling cudaTriggerProgrammaticDependentLaunch\n\n");
    for (int i = 0; i < num_iters; i++) { // run back to back iterations

       // Launch primary kernel (a different variant) followed by secondary_kernel and then another_kernel
       primary_kernel_all_blocks_skip_trigger<<<primary_kernel_blocks, primary_kernel_threads,  0, strm>>>();
       CUDA_CHECK(cudaGetLastError());

       CUDA_CHECK(cudaLaunchKernelEx(&config_secondary, secondary_kernel, d_ptr));
       CUDA_CHECK(cudaLaunchKernelEx(&config_last, another_kernel));
    }

    CUDA_CHECK(cudaStreamSynchronize(strm)); // Required

    // Cleanup resources
    CUDA_CHECK(cudaStreamDestroy(strm));
    CUDA_CHECK(cudaFree(d_ptr));
}


int main() {
    // Examples to demonstrate use of programmatic dependent launch (PDL) with CUDA streams.
    // Every example is run ITER_CNT iterations back to back.
    pdl_examples(ITER_CNT);
    return 0;
}
