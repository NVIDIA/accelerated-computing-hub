cmake_minimum_required(VERSION 3.25)
project(modern_image_processing LANGUAGES CXX CUDA)

# This example uses the CMake Package Manager (CPM) to simplify fetching CCCL from GitHub
# For more information, see https://github.com/cpm-cmake/CPM.cmake
include(cmake/CPM.cmake)

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
  NAME CCCL
  GITHUB_REPOSITORY "nvidia/cccl"
  GIT_TAG "v3.5.0.dev"
)

CPMAddPackage(
    NAME NVTX
    GITHUB_REPOSITORY NVIDIA/NVTX
    GIT_TAG v3.3.0-c-cpp
    GIT_SHALLOW TRUE)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Import OpenMP
find_package(OpenMP REQUIRED)

# Define all the executables for each step of the tutorial

add_executable(0_base_error_example 0_base_error_example.cu)
target_compile_options(0_base_error_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(0_base_error_example PUBLIC OpenMP::OpenMP_CXX PRIVATE CCCL::CCCL)

add_executable(1_span 1_span.cu)
target_compile_options(1_span PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(1_span PUBLIC OpenMP::OpenMP_CXX PRIVATE CCCL::CCCL)

add_executable(2_span_nvtx 2_span_nvtx.cu)
target_compile_options(2_span_nvtx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(2_span_nvtx PUBLIC OpenMP::OpenMP_CXX PRIVATE CCCL::CCCL nvtx3-cpp)

add_executable(3_span_nvtx_cub 3_span_nvtx_cub.cu)
target_compile_options(3_span_nvtx_cub PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(3_span_nvtx_cub PUBLIC OpenMP::OpenMP_CXX PRIVATE CCCL::CCCL nvtx3-cpp)

add_executable(4_span_nvtx_cub_container 4_span_nvtx_cub_container.cu)
target_compile_options(4_span_nvtx_cub_container PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(4_span_nvtx_cub_container PUBLIC OpenMP::OpenMP_CXX PRIVATE CCCL::CCCL nvtx3-cpp)

add_executable(5_span_nvtx_cub_container_pinned 5_span_nvtx_cub_container_pinned.cu)
target_compile_options(5_span_nvtx_cub_container_pinned PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(5_span_nvtx_cub_container_pinned PUBLIC OpenMP::OpenMP_CXX PRIVATE CCCL::CCCL nvtx3-cpp)

add_executable(6_span_nvtx_cub_container_pinned_stream 6_span_nvtx_cub_container_pinned_stream.cu)
target_compile_options(6_span_nvtx_cub_container_pinned_stream PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(6_span_nvtx_cub_container_pinned_stream PUBLIC OpenMP::OpenMP_CXX PRIVATE CCCL::CCCL nvtx3-cpp)