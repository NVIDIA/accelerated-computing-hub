cmake_minimum_required(VERSION 3.10)
project(CUDA_Samples LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 90)

file(GLOB SRC_FILES "sample*.cu")
foreach(SRC_FILE ${SRC_FILES})
    get_filename_component(EXE_NAME ${SRC_FILE} NAME_WE)
    add_executable(${EXE_NAME} ${SRC_FILE})
    set_target_properties(${EXE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endforeach()
