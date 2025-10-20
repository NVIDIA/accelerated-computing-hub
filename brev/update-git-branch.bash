#!/bin/bash

cd /accelerated-computing-hub

if [ -n "${GIT_BRANCH_NAME}" ]; then
  git checkout ${GIT_BRANCH_NAME}
fi
