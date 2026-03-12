cmake_minimum_required(VERSION 4.0)

LIST(APPEND CMAKE_PROGRAM_PATH  "/usr/local/cuda-13.1/bin")
project(cublasdx-dgemm-tutorial VERSION 0.1 LANGUAGES CUDA CXX)

# Add header tutorial helper files
add_library(tutorial_helpers INTERFACE)
target_include_directories(tutorial_helpers INTERFACE include/)

include(../cmake/common.cmake)

add_tutorial(1a_simple_dgemm_tensor src/1a_simple_dgemm_tensor.cu)
add_tutorial(1b_simple_dgemm_shared src/1b_simple_dgemm_shared.cu)
add_tutorial(1c_simple_dgemm_cublasdx src/1c_simple_dgemm_cublasdx.cu)
add_tutorial(1d_simple_pipelined_dgemm src/1d_simple_pipelined_dgemm.cu)
add_tutorial(2a_unfused_emulation src/2a_unfused_emulation/dgemm_emulation.cu)
add_tutorial(2b_partially_fused_emulation src/2b_partially_fused_emulation/dgemm_emulation.cu)
add_tutorial(2c_fully_fused_emulation src/2c_fully_fused_emulation/dgemm_emulation.cu)
add_tutorial(3a_fused_syrk_emulation src/3a_fused_syrk_emulation/syrk_emulation.cu)

