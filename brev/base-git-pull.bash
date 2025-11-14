#!/bin/bash

cd /accelerated-computing-hub

if [ -z "${ACH_DISABLE_GIT_PULL_ON_START}" ]; then
  git pull
fi
