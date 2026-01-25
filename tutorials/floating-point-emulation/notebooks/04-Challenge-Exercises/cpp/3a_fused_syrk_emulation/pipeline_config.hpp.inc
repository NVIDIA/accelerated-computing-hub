
        constexpr unsigned pipeline_depth = 2;

        auto device_pipeline = cublasdx::suggest_device_pipeline<pipeline_depth, BLAS, cublasdx::external_accumulation>(
                                   tensor_slice_a, tensor_slice_at)
                                   .value();
