#!/usr/bin/bash
# Takes 1-2 minutes to run.

# Run this script from the line_intersection directory.

# use your path with a trailing slash or leave empty to use the one on PATH
nsys_binary_dir=""
# nsys_binary_dir="/opt/nvidia/nsight-compute/2025.1.0/host/target-linux-x64/"

num_lines=100000

float_magnitudes="1e19 1e18 1e2"
impls="-DDIV -DDIV_FREE_ND -DDIV_FREE_CLRS"
bb_checks="True False"

# Iterate over every combination
test_summary=""
for impl in $impls; do
    for bb_check in $bb_checks; do

        BB_CHECK_FLAG=""
        if [ "$bb_check" == "True" ]; then
            BB_CHECK_FLAG="-DPERFORM_BB_PRECHECK"
        fi

        echo "Compiling..."
        binary="line_intersection_impl${impl}_bbCheck${bb_check}"
        nvcc --std=c++17 ${impl} ${BB_CHECK_FLAG} -lineinfo line_intersection.cu -o ${binary} -O3

        for magnitude in $float_magnitudes; do

            echo "Combination: magnitude=$magnitude, impl=$impl, bb_check=$bb_check"
            RUN_ID="num_lines${num_lines}_mag${magnitude}_impl${impl}_bbCheck${bb_check}"

            echo "Running..."
            ./${binary} 0 $num_lines 1e3 ${magnitude}

            echo "Profiling..."
            nsys_out=$(${nsys_binary_dir}nsys profile --stats true ./${binary} 0 $num_lines 1e3 ${magnitude} 2> /dev/null)
            if [ -z "$nsys_out" ]; then
                echo ""
                echo "ERROR: Could not find nsys binary install at the path specified! Kernel time will be falsely reported as zero."
                echo ""
            fi

            kernel_time_ns=$(echo "$nsys_out" | grep count_intersections_kernel | awk '{print $6}')
            kernel_time_microseconds=$((kernel_time_ns / 1000))
            echo "Kernel time: $kernel_time_microseconds microseconds."
            kernel_summary="run_id_${RUN_ID}_perf_microseconds_${kernel_time_microseconds}"
            echo "$kernel_summary"
            test_summary="${test_summary}${test_summary:+$'\n'}${kernel_summary}"
            echo ""
        done
    done
done

echo ""
echo ""

# Function to parse a line and extract fields, removing the leading "D" from Impl
parse_line() {
    local line="$1"
    # Extract fields using regex with Bash's parameter expansion
    num_lines=$(echo "$line" | grep -oP 'num_lines\K\d+')
    impl=$(echo "$line" | grep -oP 'impl-\K\S+?(?=_bbCheck)' | sed 's/^D//') # Remove leading "D"
    bb_check=$(echo "$line" | grep -oP 'bbCheck\K\S+?(?=_perf)')
    runtime=$(echo "$line" | grep -oP 'microseconds_\K\d+')

    # Output the parsed fields as a row (tab-separated for sorting)
    echo -e "$num_lines\t$bb_check\t$impl\t$runtime"
}

# Extract unique magnitudes from the input data
extract_unique_magnitudes() {
    echo "$test_summary" | grep -oP 'mag\K\S+?(?=_impl)' | sort -u
}

# Function to filter rows by magnitude and parse them
filter_rows_by_magnitude() {
    local magnitude="$1"
    echo "$test_summary" | grep "mag${magnitude}_" | while IFS= read -r line; do
        parse_line "$line"
    done
}

# Custom sort function to ensure sorting by BB_Check first, then custom Impl order
custom_sort() {
    echo -e "$1" | awk -F'\t' '
    function impl_order(impl) {
        if (impl == "DIV") return 1;
        if (impl == "DIV_FREE_ND") return 2;
        if (impl == "DIV_FREE_CLRS") return 3;
        return 4;
    }
    {
        print $2 "\t" impl_order($3) "\t" $0;
    }' | sort -k1,1 -k2,2n | cut -f3-
}

# Function to print a table from rows (with Num Lines as the first column)
print_table() {
    local rows="$1"
    echo "+-----------+----------+----------------+----------------------+"
    printf "| %-9s | %-8s | %-14s | %-20s |\n" "Num Lines" "BB_Check" "Impl" "Runtime Microseconds"
    echo "+-----------+----------+----------------+----------------------+"

    previous_bb_check=""
    
    while IFS=$'\t' read -r num_lines bb_check impl runtime; do
        # Print a horizontal line when BB_Check changes from False to True
        if [[ "$previous_bb_check" != "" && "$previous_bb_check" != "$bb_check" ]]; then
            echo "+-----------+----------+----------------+----------------------+"
        fi

        printf "| %-9s | %-8s | %-14s | %-20s |\n" "$num_lines" "$bb_check" "$impl" "$runtime"
        previous_bb_check="$bb_check"
    done <<< "$(echo -e "$rows" | sed '/^$/d')" # Remove empty lines before printing

    echo "+-----------+----------+----------------+----------------------+"
}

# Iterate over each unique magnitude and generate a table for it
generate_tables() {
    local magnitudes=$(extract_unique_magnitudes)

    for magnitude in $magnitudes; do
        rows=$(filter_rows_by_magnitude "$magnitude")
        sorted_rows=$(custom_sort "$rows")

        echo ""
        echo "Table for Float Magnitude ($magnitude):"
        print_table "$sorted_rows"
    done
}


# Generate tables for all unique magnitudes in the input data
generate_tables
