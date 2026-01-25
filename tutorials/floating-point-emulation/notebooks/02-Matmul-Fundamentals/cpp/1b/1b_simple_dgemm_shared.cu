// std libraries
#include <iostream>

// CUDA std libraries
#include <cuda/std/tuple>
#include <cuda/cmath>

// CUDA libraries
#include <cublasdx.hpp>

// utility headers
#include <tutorial_helpers.hpp>

using cublasdx::arrangement;
using size_tuple        = cuda::std::tuple<int, int, int>;
using arrangement_tuple = cuda::std::tuple<arrangement, arrangement, arrangement>;

#include "kernel.hpp.inc"

template<class Alpha, class ATensor, class BTensor, class Beta, class CTensor>
auto run_tutorial_kernel(Alpha          alpha,
                         ATensor const& tensor_a,
                         BTensor const& tensor_b,
                         Beta           beta,
                         CTensor&       tensor_c,
                         cudaStream_t   stream       = 0,
                         int            warm_up_runs = 10,
                         int            kernel_runs  = 100) {
    auto const size_m = tutorial::size<0>(tensor_a.layout());
    auto const size_n = tutorial::size<1>(tensor_b.layout());
    auto const size_k = tutorial::size<1>(tensor_a.layout());

    #include "kernel_parameters.hpp.inc"

    // 1.5 Compute remaining configuration arguments
    int const grid_dim_x = size_m / tile_m;
    int const grid_dim_y = size_n / tile_n;

    auto const block_dim = dim3(tile_m, tile_n);
    auto const grid_dim  = dim3(grid_dim_x, grid_dim_y);

    using a_value_type                    = tutorial::tensor_value_type_t<ATensor>;
    using b_value_type                    = tutorial::tensor_value_type_t<BTensor>;
    using c_value_type                    = tutorial::tensor_value_type_t<CTensor>;
    const int                 result_size = tutorial::size(tensor_c);
    std::vector<c_value_type> results(result_size);

    auto shared_memory_size = (tile_m * tile_k + tile_k * tile_n) * sizeof(double);
    auto kernel             = kernel_1b_simple_dgemm_shared<tile, ATensor, BTensor, CTensor>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));

    auto run_kernel = [&](auto& str) {
        kernel<<<grid_dim, block_dim, shared_memory_size, stream>>>(alpha, tensor_a, tensor_b, beta, tensor_c);
    };

    // correctness run
    run_kernel(stream);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    CUDA_CHECK_AND_EXIT(cudaMemcpy(results.data(),
                                   tutorial::raw_pointer_cast(tensor_c.data()),
                                   result_size * sizeof(c_value_type),
                                   cudaMemcpyDeviceToHost));

    // performance runs
    auto avg_time   = tutorial::measure::execution(run_kernel, warm_up_runs, kernel_runs, stream);
    auto avg_tflops = tutorial::real_gemm_tflops(size_m, size_n, size_k) / (avg_time);
    return cuda::std::make_tuple(avg_time, avg_tflops, results);
}

int main(int argc, char** argv) {
    // 0. Setup problem size and layout
    constexpr arrangement arr_a = arrangement::row_major;
    constexpr arrangement arr_b = arrangement::col_major;
    constexpr arrangement arr_c = arrangement::col_major;

    int const warm_up_runs = 10;
    int const kernel_runs  = 100;

    #include "parameters.hpp.inc"

    for (tutorial::gemm_problem_t problem : problems) {
        int const m = problem.m;
        int const n = problem.n;
        int const k = problem.k;
        double const alpha = problem.alpha;
        double const beta = problem.beta;

        std::cout << "Computing GEMM M=" << m << " N=" << n << " K=" << k << "\n";

        // 0.5 Setup CUDA runtime
        cudaStream_t stream;
        CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

        // 1. Generate input data

        // 2. Generate tensors with random data
        //     vector is an owning thrust::device_vector, while tensor is a view
        auto [vector_a, tensor_a] = tutorial::get_random_device_tensor<double, arr_a>(m, k);
        auto [vector_b, tensor_b] = tutorial::get_random_device_tensor<double, arr_b>(k, n);

        auto [vector_c_custom, tensor_c_custom]       = tutorial::get_random_device_tensor<double, arr_c>(m, n);
        auto [vector_c_reference, tensor_c_reference] = tutorial::get_copy_tensor(tensor_c_custom);

        // 3. Run reference
        auto [time_reference, tflops_reference, results_reference] = tutorial::cublaslt_reference(
            alpha, tensor_a, tensor_b, beta, tensor_c_reference, stream, warm_up_runs, kernel_runs);

        // 4. Run custom kernel
        auto [time_tutorial, tflops_tutorial, results_tutorial] =
            run_tutorial_kernel(alpha, tensor_a, tensor_b, beta, tensor_c_custom, stream, warm_up_runs, kernel_runs);

        // 5. Print performance and correctness summary
        std::cout << "\nCustom Kernel\n";
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
                  << "% reference performance \n";
    }

    return 0;
}
