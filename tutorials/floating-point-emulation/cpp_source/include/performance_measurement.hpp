#pragma once

#include "cuda_utilities.hpp"

namespace tutorial {

    double real_gemm_tflops(unsigned m, unsigned n, unsigned k) {
        return (2. * m * n * k) / 1e9;
    }

    double real_syrk_tflops(unsigned n, unsigned k) {
        double syrk_to_gemm_flop_ratio = ((n * (n + 1)) / 2.0) / static_cast<double>(n * n);
        return real_gemm_tflops(n, n, k) * syrk_to_gemm_flop_ratio;
    }

    struct measure {
        // Returns execution time in ms.
        template<typename Kernel>
        static float execution(Kernel&&           kernel,
                               const unsigned int warm_up_runs,
                               const unsigned int runs,
                               cudaStream_t       stream) {
            cudaEvent_t startEvent, stopEvent;
            CUDA_CHECK_AND_EXIT(cudaEventCreate(&startEvent));
            CUDA_CHECK_AND_EXIT(cudaEventCreate(&stopEvent));
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            for (unsigned int i = 0; i < warm_up_runs; i++) {
                kernel(stream);
            }

            CUDA_CHECK_AND_EXIT(cudaGetLastError());
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            CUDA_CHECK_AND_EXIT(cudaEventRecord(startEvent, stream));
            for (unsigned int i = 0; i < runs; i++) {
                kernel(stream);
            }
            CUDA_CHECK_AND_EXIT(cudaEventRecord(stopEvent, stream));
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            float time;
            CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&time, startEvent, stopEvent));
            CUDA_CHECK_AND_EXIT(cudaEventDestroy(startEvent));
            CUDA_CHECK_AND_EXIT(cudaEventDestroy(stopEvent));
            return time / runs;
        }
    };

} // namespace tutorial
