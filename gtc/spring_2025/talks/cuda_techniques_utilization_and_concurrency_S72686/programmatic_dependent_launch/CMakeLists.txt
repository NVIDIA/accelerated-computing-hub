cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(pdl_example LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 90)

add_executable(pdl_example programmatic_dependent_launch.cu)
target_include_directories(pdl_example PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_features(pdl_example PRIVATE cxx_std_17)
target_link_libraries(pdl_example PRIVATE cuda)
set_property(TARGET pdl_example PROPERTY POSITION_INDEPENDENT_CODE ON)
