#! /bin/bash

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)

DOCKER_COMPOSE=${1}
REPO=${SCRIPT_PATH}/..
MOUNT=/tmp/ach-mount

cd ${MOUNT}
docker compose -f ${DOCKER_COMPOSE} down

cd / # We've got to be somewhere that isn't the mount to unmount it.
sudo umount ${MOUNT}
