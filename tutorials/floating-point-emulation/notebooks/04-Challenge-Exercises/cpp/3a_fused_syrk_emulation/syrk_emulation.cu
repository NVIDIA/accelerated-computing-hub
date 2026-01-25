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

template<int Slices, class BLAS, class ATensor, class CTensor>
auto run_tutorial_kernel(double                alpha,
                         ATensor const&        tensor_a,
                         double                beta,
                         CTensor const&        tensor_c,
                         tutorial::matrix_half output_half,
                         cudaStream_t          stream       = 0,
                         unsigned              warm_up_runs = 10,
                         unsigned              kernel_runs  = 100,
                         bool                  debug        = false) {

    float total_time = 0.f;    

    using shift_t                = int32_t;
    using slice_value_type       = typename BLAS::a_value_type;
    using accumulator_value_type = typename BLAS::c_value_type;

    // Number of slices per elements
    auto const static_slices = cuda::std::integral_constant<int, Slices> {};

    // Create slice tensor A: [n, k, slices] - stores int8_t slices of matrix A
    auto const [shape_a_rows_, shape_a_cols_] = tensor_a.layout().shape();
    int const      shape_a_rows               = shape_a_rows_;
    int const      shape_a_cols               = shape_a_cols_;
    constexpr auto arr_a                      = cublasdx::arrangement_of_v_a<BLAS>;
    auto           d_slice_a_storage =
        tutorial::get_empty_device_tensor<slice_value_type, arr_a>(shape_a_rows, shape_a_cols, static_slices);
    // Capturing a structured binding into lambda is a C++20 feature
    auto tensor_slice_a = cuda::std::get<1>(d_slice_a_storage);

    // Construct a transposed view of A slice tensor
    auto [stride_n, stride_k, stride_slices] = tensor_slice_a.stride();
    auto const at_shape                      = cuda::std::make_tuple(shape_a_cols, shape_a_rows, static_slices);
    auto const at_stride                     = cuda::std::make_tuple(stride_k, stride_n, stride_slices);
    auto       tensor_slice_at =
        tutorial::make_gmem_tensor_from_tuples(tutorial::raw_pointer_cast(tensor_slice_a.data()), at_shape, at_stride);

    constexpr auto shift_arr = cublasdx::col_major;

    // Create tensors for the shift values with proper tiling structure
    auto const static_tile_m = cuda::std::integral_constant<int, cublasdx::size_of_v_m<BLAS>> {};
    auto       d_shift_storage =
        tutorial::get_empty_device_tensor<shift_t, shift_arr>(static_tile_m, shape_a_rows / static_tile_m());
    auto tensor_shift_a = cuda::std::get<1>(d_shift_storage);

    auto const static_tile_n   = cuda::std::integral_constant<int, cublasdx::size_of_v_n<BLAS>> {};
    auto const shift_at_shape  = cuda::std::make_tuple(static_tile_n, shape_a_rows / static_tile_n());
    auto       tensor_shift_at = tutorial::make_gmem_tensor_from_tuples<shift_arr>(
        tutorial::raw_pointer_cast(tensor_shift_a.data()), shift_at_shape);

    auto const static_tile_k   = cuda::std::integral_constant<int, cublasdx::size_of_v_k<BLAS>> {};

    // Execute preprocessing kernels to find maximum values and compute scaling factors
    {
        auto          run_preprocessing    = [&](auto str) {
            // Find max absolute value in each row of A and convert to exponent shift
            constexpr int reduction_block_size = 64;
            max_reduce_kernel<reduction_block_size, slice_matrix::a>
                <<<shape_a_rows, reduction_block_size, 0, str>>>(tensor_a, tensor_shift_a);
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
            slice_kernel<slice_kernel_block_size, Slices, slice_matrix::a>
                <<<tensor_a.size() / slice_kernel_block_size, slice_kernel_block_size, 0, str>>>(
                    tensor_a, tensor_shift_a, tensor_slice_a, shape_a_cols);
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
	    #include "pipeline_config.hpp.inc"

        #include "kernel_config.hpp.inc"

        auto dummy_c_storage = tutorial::get_copy_tensor(tensor_c);
        auto dummy_tensor_c  = cuda::std::get<1>(dummy_c_storage);

        auto run_fused_epilogue = [&](auto str) {
            kernel<<<grid, device_pipeline.get_block_dim(), shared_memory_size, str>>>(
                device_pipeline, alpha, beta, dummy_tensor_c, output_half, tensor_shift_a, tensor_shift_at);
        };

        auto time_ms = tutorial::measure::execution(run_fused_epilogue, warm_up_runs, kernel_runs, stream);
        total_time += time_ms;

        if (debug) {
            std::cout << "----> Custom Epilogue time: " << time_ms << " ms" << std::endl;
        }

        // Run correctness check
        #include "kernel_launch.hpp.inc"

        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
    }

    std::vector<double> results(tensor_c.size());
    CUDA_CHECK_AND_EXIT(cudaMemcpy(results.data(),
                                   tutorial::raw_pointer_cast(tensor_c.data()),
                                   tensor_c.size() * sizeof(double),
                                   cudaMemcpyDeviceToHost));

    // performance runs
    auto avg_tflops = tutorial::real_syrk_tflops(shape_a_rows, shape_a_cols) / total_time;
    return cuda::std::make_tuple(total_time, avg_tflops, results);
}

int main(int argc, char** argv) {
    using alpha_value_type = double;
    using beta_value_type  = double;

    constexpr auto arrangement_a = cublasdx::row_major;
    constexpr auto arrangement_c = cublasdx::col_major;

    // Automatically choose transposed
    constexpr auto arrangement_a_t = (arrangement_a == cublasdx::col_major) ? cublasdx::row_major : cublasdx::col_major;

    #include "parameters.hpp.inc"

    for (tutorial::syrk_problem_t problem : problems) {
        int const n = problem.n;
        int const k = problem.k;
        double const alpha = problem.alpha;
        double const beta = problem.beta;
        tutorial::matrix_half const output_half = problem.uplo;

        std::cout << "Computing SYRK N=" << n << " K=" << k 
            << " uplo=" << (output_half == tutorial::matrix_half::upper ? "upper" : "lower")
            << " (slices=" << slices << ")\n";
 
        // ===================================
        // Ozaki scheme configuration
        // ===================================

        #include "cublasdx_config.hpp.inc"

        if (n % tile_n != 0 or k % tile_k != 0) {
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
            n, k, range_lower_bound, range_upper_bound, seed);

        auto [vector_c_custom, tensor_c_custom] = tutorial::get_symmetric_random_device_tensor<double, arrangement_c, dist>(
            n, output_half, range_lower_bound, range_upper_bound, seed + 1);
        auto [vector_c_reference, tensor_c_reference] = tutorial::get_copy_tensor(tensor_c_custom);

        auto [time_reference, tflops_reference, results_reference] = tutorial::cublaslt_reference(
            alpha, tensor_a, beta, tensor_c_reference, output_half, stream, warm_up_runs, kernel_runs);

        auto [time_tutorial, tflops_tutorial, results_tutorial] = run_tutorial_kernel<slices, BLAS>(
            alpha, tensor_a, beta, tensor_c_custom, output_half, stream, warm_up_runs, kernel_runs);

        /* ========================================================================================= */
        /*                     Print summary of performance and correctness results */
        /* ========================================================================================= */
        std::cout << "\nCustom Emulation Kernel (fused SYRK)\n";
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
