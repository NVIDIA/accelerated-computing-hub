cmake_minimum_required(VERSION 4.0)

LIST(APPEND CMAKE_PROGRAM_PATH  "/usr/local/cuda-13.1/bin")
project(cublasdx-dgemm-notebook-1 VERSION 0.1 LANGUAGES CUDA CXX)

# Add header tutorial helper files
add_library(tutorial_helpers INTERFACE)
target_include_directories(tutorial_helpers INTERFACE ../../cpp_source/include/)

include(../../cmake/tutorial.cmake)

add_tutorial(1a_simple_dgemm_tensor cpp/1a/1a_simple_dgemm_tensor.cu)
add_tutorial(1b_simple_dgemm_shared cpp/1b/1b_simple_dgemm_shared.cu)
add_tutorial(1c_simple_pipelined_dgemm cpp/1c/1c_simple_pipelined_dgemm.cu)
add_tutorial(1d_simple_dgemm_cublasdx cpp/1d/1d_simple_dgemm_cublasdx.cu)

