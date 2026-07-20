#!/bin/bash
if [[ -n "$BASH_SOURCE" ]]; then
    SCRIPT_FILE=$(readlink -f "$BASH_SOURCE")
else
    SCRIPT_FILE=$(readlink -f "$0")
fi
source $(dirname $SCRIPT_FILE)/env.sh

bash $CARLA_ROOT/CarlaUE4.sh -quality-level=Low
