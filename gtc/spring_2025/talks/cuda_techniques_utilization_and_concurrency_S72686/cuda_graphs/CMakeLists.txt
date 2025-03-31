cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(graph_2_ways LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 90)

add_executable(graph_2_ways graph_2_ways.cu)
target_include_directories(graph_2_ways PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_features(graph_2_ways PRIVATE cxx_std_17)
target_link_libraries(graph_2_ways PRIVATE cuda)
set_property(TARGET graph_2_ways PROPERTY POSITION_INDEPENDENT_CODE ON)
