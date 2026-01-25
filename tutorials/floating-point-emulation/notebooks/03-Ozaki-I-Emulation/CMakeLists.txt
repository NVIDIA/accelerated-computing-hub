cmake_minimum_required(VERSION 4.0)

LIST(APPEND CMAKE_PROGRAM_PATH  "/usr/local/cuda-13.1/bin")
project(cublasdx-dgemm-notebook-1 VERSION 0.1 LANGUAGES CUDA CXX)

# Add header tutorial helper files
add_library(tutorial_helpers INTERFACE)
target_include_directories(tutorial_helpers INTERFACE ../../cpp_source/include/)

include(../../cmake/tutorial.cmake)

add_tutorial(2a_unfused_emulation cpp/2a_unfused_emulation/dgemm_emulation.cu)
add_tutorial(2b_partially_fused_emulation cpp/2b_partially_fused_emulation/dgemm_emulation.cu)
add_tutorial(2c_fully_fused_emulation cpp/2c_fully_fused_emulation/dgemm_emulation.cu)
