# Global CXX flags/options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
enable_testing()

# Set default arguments
set(TUTORIAL_CUDA_ARCHITECTURE "89" CACHE STRING "CUDA SM value with modifier, e.g. 89 or 100a")
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# Find cuBLASDx 
message(CHECK_START "Example Wrapper: Looking for MathDx package")
find_package(mathdx REQUIRED CONFIG
 	    PATHS
		"/opt/nvidia/mathdx/25.12"
)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED TUTORIAL_CUDA_ARCHITECTURE OR TUTORIAL_CUDA_ARCHITECTURE STREQUAL "")
	message(FATAL_ERROR "You must set TUTORIAL_CUDA_ARCHITECTURE, e.g. -DTUTORIAL_CUDA_ARCHITECTURE=89 or -DTUTORIAL_CUDA_ARCHITECTURE=90a")
endif()

if(NOT TUTORIAL_CUDA_ARCHITECTURE MATCHES "^[0-9]+[a-z]?$")
	message(FATAL_ERROR "TUTORIAL_CUDA_ARCHITECTURE must be of form sm[modifier], e.g. 89 or 100a")
endif()

string(REGEX MATCH "^([0-9]+)([A-Za-z])?$" _match "${TUTORIAL_CUDA_ARCHITECTURE}")

set(TUTORIAL_SM          "${CMAKE_MATCH_1}0")
set(TUTORIAL_SM_LETTER   "${CMAKE_MATCH_2}")  # will be empty if no letter

if(TUTORIAL_SM_LETTER STREQUAL "")
    # Case: no letter
    set(TUTORIAL_SM_MODIFIER "cublasdx::generic")

elseif(TUTORIAL_SM_LETTER STREQUAL "a")
    # Case: letter 'a'
    set(TUTORIAL_SM_MODIFIER "cublasdx::arch_specific")

elseif(TUTORIAL_SM_LETTER STREQUAL "f")
    # Case: letter 'f'
    set(TUTORIAL_SM_MODIFIER "cublasdx::family_specific")

else()
    mesage(FATAL_ERROR "Unsupported SM modifier letter '${TUTORIAL_SM_LETTER}'. Allowed: empty, 'a', or 'f'.")
endif()

set(CMAKE_CUDA_ARCHITECTURES "${TUTORIAL_CUDA_ARCHITECTURE}")

if(NOT TARGET tutorial_helpers)
    message( FATAL_ERROR "Please add tutorial_helpers library before including tutorial.cmake" )
endif()

function(add_tutorial tutorial_name tutorial_file)
    add_executable("${tutorial_name}" "${tutorial_file}")
    add_test(NAME "${tutorial_name}" COMMAND "${tutorial_name}")
    target_compile_definitions("${tutorial_name}" PUBLIC SM_VALUE=${TUTORIAL_SM})
    target_compile_definitions("${tutorial_name}" PUBLIC SM_MODIFIER_VALUE=${TUTORIAL_SM_MODIFIER})
    target_link_libraries("${tutorial_name}" PRIVATE CUDA::cublas)
    target_link_libraries("${tutorial_name}" PRIVATE mathdx::cublasdx)
    target_link_libraries("${tutorial_name}" PRIVATE tutorial_helpers)
    target_compile_options("${tutorial_name}" PRIVATE "--expt-relaxed-constexpr")
endfunction()
