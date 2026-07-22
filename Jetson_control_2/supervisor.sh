#!/usr/bin/env bash
# Runs all four pipeline stages, each in its own restart-loop, extending the
# pattern already used by control/monitoring/run_acquisition.sh. Any one
# stage crashing and restarting doesn't affect the others. Ctrl-C stops all
# four loops cleanly.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="${SCRIPT_DIR}/../Jetson_control/.venv-jellyscope/bin/python"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

run_loop() {
  local name="$1" script="$2" logfile="${LOG_DIR}/${name}.log"
  while true; do
    echo "[$(date '+%F %T')] Starting ${name}..." | tee -a "${logfile}"
    PYTHONUNBUFFERED=1 "${VENV_PY}" -u "${script}" 2>&1 | tee -a "${logfile}"
    echo "[$(date '+%F %T')] ${name} exited, restarting in 2s..." | tee -a "${logfile}"
    sleep 2
  done
}

cd "${SCRIPT_DIR}"

run_loop "record"   "${SCRIPT_DIR}/record.py"   &
run_loop "analyse"  "${SCRIPT_DIR}/analyse.py"  &
run_loop "send"     "${SCRIPT_DIR}/send.py"     &
run_loop "metadata" "${SCRIPT_DIR}/metadata.py" &

trap 'echo "Stopping all stages..."; kill $(jobs -p) 2>/dev/null' INT TERM

wait
