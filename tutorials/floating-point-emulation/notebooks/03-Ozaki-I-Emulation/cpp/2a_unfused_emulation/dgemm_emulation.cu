// std libraries
#include <iostream>

// cuda std libraries
#include <cuda/std/tuple>
#include <cuda/cmath>

// cuda libraries
#include <cublasdx.hpp>

// utility headers
#include <reference/check_error.hpp>
#include <tutorial_helpers.hpp>
#include <tensor_helpers.hpp>

#include "slicing.hpp"
#include "emulation_kernels.hpp"

// This example demonstrates the Ozaki scheme for emulating double precision GEMM
// using multiple lower precision GEMM operations. The Ozaki scheme works by:
//  1. Decomposing double precision matrices into multiple int8_t "slices"
//  2. Performing GEMM on each combination of slices
//  3. Reconstructing the final double precision result
//
// Mathematical foundation:
//   For double precision values a and b, we can represent them as:
//   a = Σ(i=0 to slices-1) a_i * 2^(shift_a - i*bits_per_slice)
//   b = Σ(j=0 to slices-1) b_j * 2^(shift_b - j*bits_per_slice)
//
//   Then a*b = ΣΣ a_i * b_j * 2^(shift_a + shift_b - (i+j)*bits_per_slice)
//
//   This allows us to compute the product using multiple int8_t GEMM operations
//   and then combine the results with appropriate scaling.

// Main cuBLASDx DGEMM emulation function using Ozaki scheme
// This function orchestrates the entire emulation process:
//   1. Preprocessing: Extract scaling factors from input matrices
//   2. Slicing: Decompose double precision matrices into int8_t slices
//   3. Matrix multiplication: Perform GEMM on slice combinations
//   4. Reconstruction: Combine results back to double precision
template<int Slices, class BLAS, class ATensor, class BTensor, class CTensor>
auto run_tutorial_kernel(double         alpha,
                         ATensor const& tensor_a,
                         BTensor const& tensor_b,
                         double         beta,
                         CTensor const& tensor_c,
                         cudaStream_t   stream       = 0,
                         unsigned       warm_up_runs = 10,
                         unsigned       kernel_runs  = 100,
                         bool           debug        = false) {

    float total_time = 0.f;

    /* ============================================================== */
    /*                    OZAKI SCHEME STEP 1: SETUP                  */
    /*                     Prepare slice tensors                      */
    /* ============================================================== */
    // Verify that tile dimensions divide evenly into problem dimensions

    // Each slice represents a portion of the original double precision values

    using slice_value_type       = typename BLAS::a_value_type;
    using accumulator_value_type = typename BLAS::c_value_type;

    // Number of slices per elements
    auto const static_slices = cuda::std::integral_constant<int, Slices> {};
    // Total number of slice matrix multiplications
    // (number of elements in diag-inclusive lower triangle of matrix with both dimensions == Slices)
    auto const static_slice_products = cuda::std::integral_constant<int, (Slices * (Slices + 1)) / 2> {};

    // Create slice tensor A: [m, k, slices] - stores int8_t slices of matrix A
    auto const [shape_a_rows_, shape_a_cols_] = tensor_a.layout().shape();
    int const      shape_a_rows               = shape_a_rows_;
    int const      shape_a_cols               = shape_a_cols_;
    constexpr auto arr_a                      = cublasdx::arrangement_of_v_a<BLAS>;
    auto           d_slice_a_storage =
        tutorial::get_empty_device_tensor<slice_value_type, arr_a>(shape_a_rows, shape_a_cols, static_slices);
    // Capturing a structured binding into lambda is a C++20 feature
    auto tensor_slice_a = cuda::std::get<1>(d_slice_a_storage);


    // Create slice tensor B: [k, n, slices] - stores int8_t slices of matrix B
    auto const [shape_b_rows_, shape_b_cols_] = tensor_b.layout().shape();
    int const      shape_b_rows               = shape_b_rows_;
    int const      shape_b_cols               = shape_b_cols_;
    constexpr auto arr_b                      = cublasdx::arrangement_of_v_b<BLAS>;
    auto           d_slice_b_storage =
        tutorial::get_empty_device_tensor<slice_value_type, arr_b>(shape_b_rows, shape_b_cols, static_slices);
    // Capturing a structured binding into lambda is a C++20 feature
    auto tensor_slice_b = cuda::std::get<1>(d_slice_b_storage);


    // Create slice tensor C: [m, n, slice_products] - stores int32_t slices of matrix C
    auto const [shape_c_rows_, shape_c_cols_] = tensor_c.layout().shape();
    int const      shape_c_rows               = shape_c_rows_;
    int const      shape_c_cols               = shape_c_cols_;
    constexpr auto arr_c                      = cublasdx::arrangement_of_v_c<BLAS>;
    auto           d_products_storage =
        tutorial::get_empty_device_tensor<accumulator_value_type, arr_c>(shape_c_rows, shape_c_cols, static_slice_products);
    // Capturing a structured binding into lambda is a C++20 feature
    auto tensor_products = cuda::std::get<1>(d_products_storage);


    /* ============================================================== */
    /*                OZAKI SCHEME STEP 2: PREPROCESSING              */
    /*           Extract max exponent of rows(A) and cols(B)          */
    /* ============================================================== */

    // The Ozaki scheme requires finding the maximum absolute value in each
    // row of A and each column of B to determine appropriate scaling factors.
    // These scaling factors ensure that when we slice the double precision
    // values into int8_t components, we don't lose significant precision.

    using shift_t            = int32_t;
    constexpr auto shift_arr = cublasdx::col_major;

    // Create tensors for the shift values with proper tiling structure
    auto const static_tile_m = cuda::std::integral_constant<int, cublasdx::size_of_v_m<BLAS>> {};
    auto       d_shift_a_storage =
        tutorial::get_empty_device_tensor<shift_t, shift_arr>(static_tile_m, shape_a_rows / static_tile_m());
    auto tensor_shift_a = cuda::std::get<1>(d_shift_a_storage);

    auto const static_tile_n = cuda::std::integral_constant<int, cublasdx::size_of_v_n<BLAS>> {};
    auto       d_shift_b_storage =
        tutorial::get_empty_device_tensor<shift_t, shift_arr>(static_tile_n, shape_b_cols / static_tile_n());
    auto tensor_shift_b = cuda::std::get<1>(d_shift_b_storage);

    // Execute preprocessing kernels to find maximum values and compute scaling factors
    {
        auto          run_preprocessing    = [&](auto str) {
            constexpr int reduction_block_size = 64;
            // Find max absolute value in each row of A and convert to exponent shift
            max_reduce_kernel<reduction_block_size, slice_matrix::a>
                <<<shape_a_rows, reduction_block_size, 0, str>>>(tensor_a, tensor_shift_a);
            // Find max absolute value in each column of B and convert to exponent shift
            max_reduce_kernel<reduction_block_size, slice_matrix::b>
                <<<shape_b_cols, reduction_block_size, 0, str>>>(tensor_b, tensor_shift_b);
        };

        auto time_ms = tutorial::measure::execution(run_preprocessing, warm_up_runs, kernel_runs, stream);

        total_time += time_ms;
        if (debug) {
            std::cout << "----> Custom Preprocess time: " << time_ms << " ms" << std::endl;
        }

        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
    }

    /* ============================================================== */
    /*                    OZAKI SCHEME STEP 3: SLICING                */
    /*                  Slice up input A and B matrices               */
    /* ============================================================== */

    // This step decomposes each double precision value into multiple int8_t slices.
    // For a double precision value x with scaling factor s, we create slices such that:
    //   x ≈ Σ(i=0 to slices-1) slice_i * 2^(s - i*8)
    // where each slice_i is an int8_t value.

    {

        auto run_slicing = [&](auto str) {
            constexpr auto slice_kernel_block_size = 64;
            // Slice matrix A: each double precision element becomes slices int8_t values
            slice_kernel<slice_kernel_block_size, Slices, slice_matrix::a>
                <<<tensor_a.size() / slice_kernel_block_size, slice_kernel_block_size, 0, str>>>(
                    tensor_a, tensor_shift_a, tensor_slice_a, shape_a_cols);
            // Slice matrix B: each double precision element becomes slices int8_t values
            slice_kernel<slice_kernel_block_size, Slices, slice_matrix::b>
                <<<tensor_b.size() / slice_kernel_block_size, slice_kernel_block_size, 0, str>>>(
                    tensor_b, tensor_shift_b, tensor_slice_b, shape_a_cols);
        };

        auto time_ms = tutorial::measure::execution(run_slicing, warm_up_runs, kernel_runs, stream);
        total_time += time_ms;

        if (debug) {
            std::cout << "----> Custom Slice time: " << time_ms << " ms" << std::endl;
        }

        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
    }

    /* ============================================================== */
    /*            OZAKI SCHEME STEP 4: MATRIX MULTIPLICATION          */
    /*                      Product of slices                         */
    /* ============================================================== */

    // This is the core of the Ozaki scheme. We need to compute the product:
    //   C = A * B = (Σ A_i * 2^shift_A_i) * (Σ B_j * 2^shift_B_j)
    //     = ΣΣ A_i * B_j * 2^(shift_A_i + shift_B_j)
    //
    // We compute this as multiple GEMM operations between slice combinations,
    // with each result scaled appropriately and accumulated into the final result.

    {
        auto run_unfused_matmul = [&](auto str) {
	    #include "slice_coordination.hpp.inc"
        };

        auto time_ms = tutorial::measure::execution(run_unfused_matmul, warm_up_runs, kernel_runs, stream);
        total_time += time_ms;

        if (debug) {
            std::cout << "----> Custom Matmul time: " << time_ms << " ms" << std::endl;
        }

        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
    }

    /* ============================================================== */
    /*            OZAKI SCHEME STEP 5: EPILOGUE                       */
    /*                      Accumulate Diagonals                      */
    /* ============================================================== */

    {
        #include "epilogue_config.hpp.inc"

        dim3 grid(shape_a_rows / epilogue_kernel_tile_m, shape_b_cols / epilogue_kernel_tile_n);
        dim3 block(epilogue_kernel_tile_m, epilogue_kernel_tile_n);

        auto dummy_c_storage = tutorial::get_copy_tensor(tensor_c);
        auto dummy_tensor_c  = cuda::std::get<1>(dummy_c_storage);

        auto run_epilogue = [&](auto str) {
            epilogue_kernel<epilogue_kernel_tile_m * epilogue_kernel_tile_n, Slices>
                <<<grid, block, 0, str>>>(alpha, beta, tensor_products, tensor_shift_a, tensor_shift_b, dummy_tensor_c);
        };

        auto time_ms = tutorial::measure::execution(run_epilogue, warm_up_runs, kernel_runs, stream);
        total_time += time_ms;

        if (debug) {
            std::cout << "----> Custom Epilogue time: " << time_ms << " ms" << std::endl;
        }

        // Run correctness check
        epilogue_kernel<epilogue_kernel_tile_m * epilogue_kernel_tile_n, Slices>
            <<<grid, block, 0, stream>>>(alpha, beta, tensor_products, tensor_shift_a, tensor_shift_b, tensor_c);

        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
    }

    std::vector<double> results(tensor_c.size());
    CUDA_CHECK_AND_EXIT(cudaMemcpy(results.data(),
                                   tutorial::raw_pointer_cast(tensor_c.data()),
                                   tensor_c.size() * sizeof(double),
                                   cudaMemcpyDeviceToHost));

    // performance runs
    auto avg_tflops = tutorial::real_gemm_tflops(shape_a_rows, shape_b_cols, shape_a_cols) / total_time;
    return cuda::std::make_tuple(total_time, avg_tflops, results);
}

int main(int argc, char** argv) {
    using alpha_value_type = double;
    using beta_value_type  = double;

    constexpr auto arrangement_a = cublasdx::row_major;
    constexpr auto arrangement_b = cublasdx::col_major;
    constexpr auto arrangement_c = cublasdx::col_major;

    int const warm_up_runs = 10;
    int const kernel_runs = 100;

    #include "parameters.hpp.inc"

    bool const debug = false;

    for (tutorial::gemm_problem_t problem : problems) {
        int const m = problem.m;
        int const n = problem.n;
        int const k = problem.k;
        double const alpha = problem.alpha;
        double const beta = problem.beta;

        std::cout << "Computing GEMM M=" << m << " N=" << n << " K=" << k << " (slices=" << slices << ")\n";

        // ===================================
        // Ozaki scheme configuration
        // ===================================

        #include "cublasdx_config.hpp.inc"

        if (m % tile_m != 0 or n % tile_n != 0 or k % tile_k != 0) {
            std::cerr << "Problem shape must be divisible by tile shape" << std::endl;
            exit(-1);
        }

        // ===================================
        // Data type definitions
        // ===================================

        using a_value_type = double;
        using b_value_type = double;
        using c_value_type = double;

        cudaStream_t stream;
        CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

        if (debug) {
            tutorial::print_device_properties();
        }

        /* ============================================================== */
        /*                     Input FP64 (host) tensors                  */
        /* ============================================================== */
        static const float                      range_lower_bound = 1.0f / 3.14f;
        static const float                      range_upper_bound = 52.0f / 3.14f;
        int                                     seed              = 1234;
        constexpr tutorial::random_distribution dist              = tutorial::random_distribution::uniform;

        auto [vector_a, tensor_a] = tutorial::get_random_device_tensor<double, arrangement_a, dist>(
            m, k, range_lower_bound, range_upper_bound, seed);
        auto [vector_b, tensor_b] = tutorial::get_random_device_tensor<double, arrangement_b, dist>(
            k, n, range_lower_bound, range_upper_bound, seed + 1);

        auto [vector_c_custom, tensor_c_custom] = tutorial::get_random_device_tensor<double, arrangement_c, dist>(
            m, n, range_lower_bound, range_upper_bound, seed + 2);
        auto [vector_c_reference, tensor_c_reference] = tutorial::get_copy_tensor(tensor_c_custom);

        /* ============================================================== */
        /*                       Compute Reference Result                 */
        /* ============================================================== */
        auto [time_reference, tflops_reference, results_reference] = tutorial::cublaslt_reference(
            alpha, tensor_a, tensor_b, beta, tensor_c_reference, stream, warm_up_runs, kernel_runs);


        /* ============================================================== */
        /*                     Compute Emulation Result                   */
        /* ============================================================== */
        auto [time_tutorial, tflops_tutorial, results_tutorial] = run_tutorial_kernel<slices, BLAS>(
            alpha, tensor_a, tensor_b, beta, tensor_c_custom, stream, warm_up_runs, kernel_runs, debug);

        /* ========================================================================================= */
        /*                     Print summary of performance and correctness results */
        /* ========================================================================================= */
        std::cout << "\nCustom Emulation Kernel (unfused)\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Avg time [ms]  = " << time_tutorial << "\n";
        std::cout << "Avg TFLOP/s  = " << tflops_tutorial << "\n";

        std::cout << "\ncuBLASLt (not including heuristic)\n";
        std::cout << "Avg time [ms]  = " << time_reference << "\n";
        std::cout << "Avg TFLOP/s  = " << tflops_reference << "\n\n";

        constexpr bool verbose_knob = false;
        constexpr bool print_knob   = true;

        auto error = tutorial::calculate_error(results_tutorial, results_reference, verbose_knob, print_knob);
        std::cout << std::fixed << std::setprecision(10) << "Total relative error = " << error << "\n";

        std::cout << std::fixed << std::setprecision(2) << (tflops_tutorial / tflops_reference) * 100
                  << "% reference performance \n\n";
    }

    return 0;
}
