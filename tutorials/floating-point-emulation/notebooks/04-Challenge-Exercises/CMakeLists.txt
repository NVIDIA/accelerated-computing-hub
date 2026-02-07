cmake_minimum_required(VERSION 4.0)

LIST(APPEND CMAKE_PROGRAM_PATH  "/usr/local/cuda-13.1/bin")
project(cublasdx-dgemm-notebook-3 VERSION 0.1 LANGUAGES CUDA CXX)

# Add header tutorial helper files
add_library(tutorial_helpers INTERFACE)
target_include_directories(tutorial_helpers INTERFACE ../../cpp_source/include/)

include(../../cmake/tutorial.cmake)

add_tutorial(3a_fused_syrk_emulation cpp/3a_fused_syrk_emulation/syrk_emulation.cu)
