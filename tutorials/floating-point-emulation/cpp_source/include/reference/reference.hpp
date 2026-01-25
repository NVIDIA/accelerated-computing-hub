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

#pragma once

#include "cublaslt_runner.hpp"
#include "check_error.hpp"
#include <performance_measurement.hpp>

namespace tutorial {

    template<class Alpha, class ATensor, class BTensor, class Beta, class CTensor>
    auto cublaslt_reference(Alpha          alpha,
                            ATensor const& tensor_a,
                            BTensor const& tensor_b,
                            Beta           beta,
                            CTensor&       tensor_c,
                            cudaStream_t   stream                 = 0,
                            int            kernel_warm_up_repeats = 10,
                            int            kernel_repeats         = 100) {

        using cublas_a_value_type = tensor_value_type_t<ATensor>;
        using cublas_b_value_type = tensor_value_type_t<BTensor>;
        using cublas_c_value_type = tensor_value_type_t<CTensor>;

        assert(cute::size<0>(tensor_a) == cute::size<0>(tensor_c)); // Check M
        assert(cute::size<1>(tensor_b) == cute::size<1>(tensor_c)); // Check N
        assert(cute::size<1>(tensor_a) == cute::size<0>(tensor_b)); // Check K

        int size_m = cute::size<0>(tensor_a);
        int size_n = cute::size<1>(tensor_b);
        int size_k = cute::size<1>(tensor_a);

        auto global_shape = cute::make_tuple(size_m, size_n, size_k);

        assert(cute::stride<0>(tensor_a) == 1 or cute::stride<1>(tensor_a) == 1); // Verify if A is either Row/Col major
        assert(cute::stride<0>(tensor_b) == 1 or cute::stride<1>(tensor_b) == 1); // Verify if B is either Row/Col major
        assert(cute::stride<0>(tensor_c) == 1 or cute::stride<1>(tensor_c) == 1); // Verify if C is either Row/Col major

        auto arr_a =
            (cute::stride<0>(tensor_a) == 1) ? cublasdx::arrangement::col_major : cublasdx::arrangement::row_major;
        auto arr_b =
            (cute::stride<0>(tensor_b) == 1) ? cublasdx::arrangement::col_major : cublasdx::arrangement::row_major;
        auto arr_c =
            (cute::stride<0>(tensor_c) == 1) ? cublasdx::arrangement::col_major : cublasdx::arrangement::row_major;
        auto global_arrangement = cute::make_tuple(arr_a, arr_b, arr_c);

        auto [time_cublas, results_cublas] =
            example::cublaslt_runner<cublas_a_value_type, cublas_b_value_type, cublas_c_value_type>(global_shape,
                                                                                                    global_arrangement)
                .execute_with_time_and_results(alpha,
                                               cute::raw_pointer_cast(tensor_a.data()),
                                               cute::raw_pointer_cast(tensor_b.data()),
                                               beta,
                                               cute::raw_pointer_cast(tensor_c.data()),
                                               kernel_warm_up_repeats,
                                               kernel_repeats,
                                               stream);

        auto tflops_cublas = real_gemm_tflops(size_m, size_n, size_k) / time_cublas;

        return cuda::std::make_tuple(time_cublas, tflops_cublas, results_cublas);
    }

    template<class ATensor, class CTensor>
    auto cublaslt_reference(double         alpha,
                            ATensor const& tensor_a,
                            double         beta,
                            CTensor&       tensor_c,
                            matrix_half    output_half,
                            cudaStream_t   stream                 = 0,
                            int            kernel_warm_up_repeats = 10,
                            int            kernel_repeats         = 100) {
        // Convert arguments to cuBLAS format
        //
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetStream(handle, stream);

        int n = cute::get<0>(tensor_a.layout().shape());
        int k = cute::get<1>(tensor_a.layout().shape());

        bool const is_c_row_major        = cute::get<1>(tensor_c.stride()) == 1;
        auto const reversed_output_half  = output_half == matrix_half::lower ? matrix_half::upper : matrix_half::lower;
        auto const effective_output_half = is_c_row_major ? reversed_output_half : output_half;
        auto const cublas_fill_mode =
            (effective_output_half == matrix_half::lower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
        auto const cublas_a_trans = (cute::get<0>(tensor_a.stride()) == 1) ? CUBLAS_OP_N : CUBLAS_OP_T;
        auto const cublas_lda =
            (cublas_a_trans == CUBLAS_OP_N) ? cute::get<0>(tensor_a.shape()) : cute::get<1>(tensor_a.shape());
        auto const cublas_ldc = cute::get<0>(tensor_c.layout().shape());

        auto run_cublas = [&](cudaStream_t&) {
            cublasDsyrk(handle,
                        cublas_fill_mode,
                        cublas_a_trans,
                        n,
                        k,
                        &alpha,
                        raw_pointer_cast(tensor_a.data()),
                        cublas_lda,
                        &beta,
                        raw_pointer_cast(tensor_c.data()),
                        cublas_ldc);
        };

        // Run cuBLAS for correctness
        run_cublas(stream);
        // Copy results
        std::vector<double> results(tensor_c.size());
        CUDA_CHECK_AND_EXIT(cudaMemcpy(results.data(),
                                       raw_pointer_cast(tensor_c.data()),
                                       tensor_c.size() * sizeof(double),
                                       cudaMemcpyDeviceToHost));

        // Measure performance
        auto time_cublas   = measure::execution(run_cublas, kernel_warm_up_repeats, kernel_repeats, stream);
        auto tflops_cublas = real_syrk_tflops(n, k) / time_cublas;

        // Clean up and return
        cublasDestroy(handle);
        return cuda::std::make_tuple(time_cublas, tflops_cublas, results);
    }
} // namespace tutorial
