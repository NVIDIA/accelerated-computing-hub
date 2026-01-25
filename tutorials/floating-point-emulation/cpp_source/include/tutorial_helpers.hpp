#pragma once

#include <random>
#include <complex>
#include <sstream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cublasdx.hpp>

// Utilities for Random Number Generation
#include "random.hpp"
// Utilities for numerical conversions
#include "numerical.hpp"
// Utilities for stable performance measurement
#include "performance_measurement.hpp"
// Helpers types and utilities for matmul performance
#include "performance_measurement.hpp"
// Helpers for CUDA Runtime
#include "cuda_utilities.hpp"
// Intermediate layer for managing layouts and tensors
#include "tensor_helpers.hpp"
// cuBLASLt reference with performance autotuning
#include "reference/reference.hpp"

namespace tutorial {

    struct gemm_problem_t {
        int m;
        int n;
        int k;
        double alpha;
        double beta;
    };

    struct syrk_problem_t {
        int n;
        int k;
        double alpha;
        double beta;
        matrix_half uplo;
    };

    void print_device_properties() {
        cudaDeviceProp prop;
        int            sm_clock, mem_clock;

        int device_count = 0;
        CUDA_CHECK_AND_EXIT(cudaGetDeviceCount(&device_count));

        std::stringstream ss;
        ss << "Number of CUDA devices: " << device_count << std::endl << std::endl;

        for (auto device_id = 0; device_id < device_count; device_id++) {
            CUDA_CHECK_AND_EXIT(cudaGetDeviceProperties(&prop, device_id));
            CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&sm_clock, cudaDevAttrClockRate, device_id));
            CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&mem_clock, cudaDevAttrMemoryClockRate, device_id));

            ss << "Device " << device_id << ": " << prop.name << std::endl;
            ss << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
            ss << "  Total global memory: " << (prop.totalGlobalMem >> 20) << " MB" << std::endl;
            ss << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
            ss << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
            ss << "  Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
            ss << "  Warp size: " << prop.warpSize << std::endl;

            ss << "  Clock Rate: " << sm_clock / 1000.f << " MHz" << std::endl;
            ss << "  Memory Clock Rate: " << mem_clock / 1000.f << " MHz" << std::endl;

            ss << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
            ss << std::endl;
        }

        std::cout << ss.str();
    }

} // namespace tutorial
