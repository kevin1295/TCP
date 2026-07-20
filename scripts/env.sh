#!/bin/bash
if [[ -n "$BASH_SOURCE" ]]; then
    SCRIPT_FILE=$(readlink -f "$BASH_SOURCE")
else
    SCRIPT_FILE=$(readlink -f "$0")
fi
echo "Setting Workspace Root: $(dirname $(dirname $SCRIPT_FILE))"
export WORKSPACE_ROOT=$(dirname $(dirname $SCRIPT_FILE))

export CARLA_ROOT=~/carla_0.9.15

export SCENARIO_RUNNER_ROOT=${WORKSPACE_ROOT}/scenario_runner
export LEADERBOARD_ROOT=${WORKSPACE_ROOT}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"

source ${WORKSPACE_ROOT}/ros_ws/install/setup.bash
