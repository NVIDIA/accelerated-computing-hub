#! /bin/bash

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)

DOCKER_COMPOSE=${1}
REPO=${SCRIPT_PATH}/..
MOUNT=/tmp/ach-mount

sudo mkdir -p ${MOUNT}
sudo bindfs --force-user=$(id -u) --force-group=$(id -g) \
            --create-for-user=$(id -u) --create-for-group=$(id -g) \
            ${REPO} ${MOUNT}

cd ${MOUNT}
docker compose -f ${DOCKER_COMPOSE} up -d
