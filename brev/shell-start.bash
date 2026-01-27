#! /bin/bash
#
# Entrypoint for interactive shell sessions.

source /accelerated-computing-hub/brev/dev-common.bash
create_user_and_switch login

exec bash -l
