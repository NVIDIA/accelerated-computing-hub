#!/bin/bash

cd /accelerated-computing-hub

if [ ! -f /accelerated-computing-hub/.git-checkout-done ]; then
    git checkout ${GIT_BRANCH_NAME:-main}
    touch /accelerated-computing-hub/.git-checkout-done
fi

exec "$@"
