
        dim3      grid(shape_a_rows / static_tile_m(), shape_a_rows / static_tile_n());
        auto kernel = fused_epilogue_kernel<BLAS,
                                            decltype(device_pipeline),
                                            double,
                                            double,
                                            CTensor,
                                            decltype(tensor_shift_a),
                                            decltype(tensor_shift_at),
                                            Slices>;

        auto shared_memory_size = cublasdx::make_shared_storage_calculator()
                                      .add(device_pipeline.buffer_alignment(), device_pipeline.buffer_size())
                                      .add(16, sizeof(int32_t), static_tile_m()) // shift_a
                                      .add(16, sizeof(int32_t), static_tile_n()) // shift_b
                                      .get();

        CUDA_CHECK_AND_EXIT(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
