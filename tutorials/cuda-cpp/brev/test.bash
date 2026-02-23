#! /bin/bash

START_TIME=$(date +%s.%N)

nvidia-smi

END_TIME=$(date +%s.%N)
ELAPSED=$(awk "BEGIN {print $END_TIME - $START_TIME}")

echo ""
awk -v elapsed="$ELAPSED" 'BEGIN {
    hours = int(elapsed / 3600)
    minutes = int((elapsed % 3600) / 60)
    seconds = elapsed % 60
    printf "Elapsed time: %dh %dm %.3fs\n", hours, minutes, seconds
}'

exit 0
