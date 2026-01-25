
template<class BLAS,
         class DevicePipeline,
         class Alpha,
         class Beta,
         class CTensor,
         class AShiftTensor,
         class AtShiftTensor,
         int32_t Slices>
__launch_bounds__(DevicePipeline::max_threads_per_block, 1) __global__
    void fused_epilogue_kernel(__grid_constant__ DevicePipeline const device_pipeline,
                               Alpha                                  alpha,
                               Beta                                   beta,
                               CTensor                                gmem_c_fp64,
                               tutorial::matrix_half                  output_half,
                               AShiftTensor const                     gmem_shift_a,
                               AtShiftTensor const                    gmem_shift_at) {
    extern __shared__ __align__(device_pipeline.buffer_alignment()) char smem[];
#ifdef __CUDA_ARCH__
    if constexpr (cublasdx::sm_of_v<BLAS> == __CUDA_ARCH__) {
        // ================================
        // 1. SETUP AND TILE PREPARATION
        // ================================

        constexpr int tile_m = cublasdx::size_of_v_m<BLAS>;
        constexpr int tile_n = cublasdx::size_of_v_n<BLAS>;

        auto const block_offset_m = blockIdx.x * tile_m;
        auto const block_offset_n = blockIdx.y * tile_n;

        if ((block_offset_n > (block_offset_m + tile_m) and output_half == tutorial::matrix_half::lower) or
            (block_offset_m > (block_offset_n + tile_n) and output_half == tutorial::matrix_half::upper)) {
            return;
        }

        constexpr auto initial_diag = Slices - 1;
        constexpr auto initial_term = 0;

        auto [pipeline_smem, smem_shift_a, smem_shift_at] =
            cublasdx::shared_memory::slice<char, int32_t, int32_t>(smem,
                                                                   device_pipeline.buffer_alignment(),
                                                                   device_pipeline.buffer_size(),
                                                                   cublasdx::alignment_of_v_a<BLAS>,
                                                                   cute::make_layout(cute::Int<tile_m>()),
                                                                   cublasdx::alignment_of_v_b<BLAS>,
                                                                   cute::make_layout(cute::Int<tile_n>()));

        // Copy general purpose data
        cublasdx::copy<BLAS, 16>(gmem_shift_a(cute::_, blockIdx.x), smem_shift_a);
        cublasdx::copy<BLAS, 16>(gmem_shift_at(cute::_, blockIdx.y), smem_shift_at);
        cublasdx::copy_wait();


        // Get pipeline tile
        auto tile_pipeline = device_pipeline.get_tile(pipeline_smem,
                                                      cublasdx::make_coord(blockIdx.x, initial_term),
                                                      cublasdx::make_coord(blockIdx.y, initial_diag));

        auto accumulator = tile_pipeline.get_accumulator();

        // ================================
        // 2. FP64 C INPUT / OUTPUT TILE SETUP
        // ================================

        auto tile_c_fp64_gmem = cublasdx::get_tile(gmem_c_fp64, BLAS::c_shape, blockIdx.x, blockIdx.y);

        // ============================================
        // 3. OZAKI SCHEME DIAGONAL ITERATION
        // ============================================

        // Iterate over diagonals in reverse order (highest power of 2 first)
        // This ensures proper accumulation order for numerical stability
#    pragma unroll 1
        for (auto diag = initial_diag; diag >= 0; --diag) {

            // Initialize accumulator for this diagonal
            accumulator.clear();

            // ==========================================
            // 4. SLICE COMBINATION COMPUTATION
            // ==========================================

            // Compute all slice combinations that contribute to this diagonal
            // For diagonal d, we compute: A_slice[i] * B_slice[d-i] for i = 0 to d
#    pragma unroll 1
            for (auto term = initial_term; term <= diag; ++term) {
                // =========================================
                // 5. N-STAGE MEMORY PIPELINE FOR GEMM
                // =========================================

                tile_pipeline.execute(accumulator);

                const auto next_slice_row = (term == diag) ? 0 : term + 1;                         // A slice index
                const auto next_slice_col = (term == diag) ? (diag - 1) : (diag - next_slice_row); // B slice index
                device_pipeline.reset_tile(tile_pipeline,
                                           cublasdx::make_coord(blockIdx.x, next_slice_row),
                                           cublasdx::make_coord(blockIdx.y, next_slice_col));
            } /* end of slice combination loop */

            // ========================================
            // 6. RESULT RECONSTRUCTION AND EPILOGUE
            // ========================================


            // Load existing C values
            auto c_fp64_frag = accumulator.make_partition_and_copy(tile_c_fp64_gmem);

            if(accumulator.is_thread_active()) {
                // Convert accumulated int32_t results back to double precision
                // and apply appropriate scaling based on slice positions
                auto gemm_results = accumulator.get_results();


                // Process each element in the register fragment
#    pragma unroll
                for (int i = 0; i < cublasdx::size(c_fp64_frag); ++i) {
                    const auto [global_x, global_y] = accumulator.map_fragment_index(i);
                    const auto shift_a_elem         = smem_shift_a(global_x);
                    const auto shift_at_elem        = smem_shift_at(global_y);

                    auto const total_x = block_offset_m + global_x;
                    auto const total_y = block_offset_n + global_y;
                    bool const is_in_bounds = (output_half == tutorial::matrix_half::lower and (total_x >= total_y)) or
                                              (output_half == tutorial::matrix_half::upper and (total_y >= total_x));

                    // Convert int32_t slice result back to double precision
                    // with appropriate scaling for this diagonal and element
                    double const scaled_unsliced_value =
                        alpha * nth_slice_to_fp64<int32_t, int8_t>(diag, gemm_results(i), shift_a_elem + shift_at_elem);

                    c_fp64_frag(i) = is_in_bounds ? (scaled_unsliced_value + (diag == initial_diag ? beta : 1.0) * c_fp64_frag(i)) : c_fp64_frag(i);
                }

            }

            accumulator.partition_and_copy(c_fp64_frag, tile_c_fp64_gmem);
        }

    }
#endif
}
