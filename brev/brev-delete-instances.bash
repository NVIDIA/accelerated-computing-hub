#!/bin/bash

INSTANCES=$(brev list | tail -n +3 | awk '{print $1}' | tr '\n' ' ')

if [ -n "${INSTANCES}" ]; then
    brev delete ${INSTANCES}
else
    echo "No instances found"
fi
