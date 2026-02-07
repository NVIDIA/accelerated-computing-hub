
        kernel<<<grid, device_pipeline.get_block_dim(), shared_memory_size, stream>>>(
            device_pipeline, alpha, beta, tensor_c, output_half, tensor_shift_a, tensor_shift_at);
