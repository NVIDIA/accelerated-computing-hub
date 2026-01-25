def _monkey_patch_cublasdx_numba():
    # WAR: (will be fixed int 0.8.1+)
    from nvmath.device import cublasdx_numba as _cublasdx_numba_monkey_patch
    if "with_pipeline" not in _cublasdx_numba_monkey_patch._BLAS_DEFINITION_ARGS:
        _cublasdx_numba_monkey_patch._BLAS_DEFINITION_ARGS += [
            "with_pipeline", "enable_input_streaming", "static_block_dim"
        ]

_monkey_patch_cublasdx_numba()
