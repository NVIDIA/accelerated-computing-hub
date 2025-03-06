#!/usr/bin/bash

# Run this script from the line_intersection directory.

# ========== Configuration =================
# Set all these
manual_run_id="my_descriptive_run_name"
num_lines=100000

run_nsys="True"
# use your path with a trailing slash or leave empty to use the one on PATH
nsys_binary_dir=""
# nsys_binary_dir="/opt/nvidia/nsight-compute/2025.1.0/host/target-linux-x64/"

run_ncu="False" # Not enabled on launchable
# use your path with a trailing slash or leave empty to use the one on PATH
ncu_binary_dir=""

# pick one
impl="-DDIV"
# impl="-DDIV_FREE_ND"
# impl="-DDIV_FREE_CLRS"

# pick one
magnitude="1e19"
# magnitude="1e18"
# magnitude="1e2"

# pick one
# bb_check="True"
bb_check="False"
# ========== Configuration =================


BB_CHECK_FLAG=""
if [ "$bb_check" == "True" ]; then
    BB_CHECK_FLAG="-DPERFORM_BB_PRECHECK"
fi

echo "Combination: magnitude=$magnitude, impl=$impl, bb_check=$bb_check"

echo "Compiling..."
binary="line_intersection_impl${impl}_bbCheck${bb_check}"
nvcc --std=c++17 ${impl} ${BB_CHECK_FLAG} -lineinfo line_intersection.cu -o ${binary} -O3

echo "Running..."
RUN_ID="${manual_run_id}_num_lines${num_lines}_mag${magnitude}_impl${impl}_bbCheck${bb_check}"
./${binary} 0 $num_lines 1e3 ${magnitude}

if [ "$run_nsys" == "True" ]; then
    echo "Nsys Profiling..."
    nsys_out=$(${nsys_binary_dir}nsys profile -o ${RUN_ID} --stats true ./${binary} 0 $num_lines 1e3 ${magnitude} 2> /dev/null)
    if [ -z "$nsys_out" ]; then
        echo ""
        echo "ERROR: Could not find nsys binary install at the path specified! Kernel time will be falsely reported as zero."
        echo ""
    fi
    kernel_time_ns=$(echo "$nsys_out" | grep count_intersections_kernel | awk '{print $6}')
    kernel_time_microseconds=$((kernel_time_ns / 1000))
    echo "Kernel time: $kernel_time_microseconds microseconds."
fi

if [ "$run_ncu" == "True" ]; then

    echo "Ncu Profiling..."
    ${ncu_binary_dir}ncu -f --set full --import-source on -o ${RUN_ID}.ncu-rep \
        --target-processes all \
        -k count_intersections_kernel \
        ./${binary} 0 $num_lines 1e3 ${magnitude}
fi

