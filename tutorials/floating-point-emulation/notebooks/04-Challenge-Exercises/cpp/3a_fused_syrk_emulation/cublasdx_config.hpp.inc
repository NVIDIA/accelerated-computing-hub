
    using slice_value_type       = int8_t;  // Precision for individual slices
    using accumulator_value_type = int32_t; // Precision for accumulation

    // The shape of data tile processed by a single CTA block
    constexpr int tile_m = 128;
    constexpr int tile_n = 128;
    constexpr int tile_k = 128;

    // The shape of CTA block (number of threads)
    constexpr int cta_shape_x = 256;
    constexpr int cta_shape_y = 1;
    constexpr int cta_shape_z = 1;

    using BLAS = decltype(cublasdx::Size<tile_m, tile_n, tile_k>() +
                          cublasdx::Precision<slice_value_type, slice_value_type, accumulator_value_type>() +
                          cublasdx::Type<cublasdx::type::real>() + cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<arrangement_a, arrangement_a_t, arrangement_c>() + cublasdx::Block() +
                          cublasdx::BlockDim<cta_shape_x, cta_shape_y, cta_shape_z>() + cublasdx::StaticBlockDim() +
                          cublasdx::MaxAlignment() + cublasdx::EnableInputStreaming() + cublasdx::WithPipeline() +
                          cublasdx::SM<SM_VALUE, SM_MODIFIER_VALUE>());
