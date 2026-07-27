#!/usr/bin/env bash
# Runs all four pipeline stages, each in its own restart-loop, extending the
# pattern already used by control/monitoring/run_acquisition.sh. Any one
# stage crashing and restarting doesn't affect the others. Ctrl-C stops all
# four loops cleanly.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Self-contained venv living inside Jetson_monitoring/ itself (one level up from
# Monitor/) -- no longer borrowed from Jetson_control/, which doesn't have one anymore.
VENV_PY="${SCRIPT_DIR}/../.venv-jellyscope_on_jetson/bin/python"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

run_loop() {
  # NOT `local name="$1" logfile="...${name}..."` on one line -- bash expands every
  # RHS in a compound `local` statement before any of them actually become local
  # variables, so `${name}` there gets read while `name` doesn't exist yet, which
  # `set -u` (correctly) flags as unbound. Sequential statements avoid that.
  local name="$1" script="$2"
  local logfile="${LOG_DIR}/${name}.log"
  while true; do
    # Straight file redirection, not `| tee -a` to this terminal -- piping through tee
    # ties the pipeline's liveness to this terminal/SSH session. If that connection goes
    # half-dead (network black-holed, not a clean close) tee's write() to it blocks
    # forever; once the resulting backpressure fills the 64KB pipe buffer back to
    # python's own print(), the whole process blocks on its next print() call and never
    # returns, so it never exits and this loop's restart never fires -- a silent,
    # permanent hang. Root cause of the 2026-07-25 freeze -- analyse.py deadlocked at
    # frame 5673 (its higher per-frame log volume filled the pipe buffer faster),
    # record.py followed ~3h later at frame 16490, and the login session (whose SSH
    # connection triggered this) lingered for 5.5h waiting for both to exit.
    # `>>` to a plain file has no such reader to block on. Use `tail -f logs/<name>.log`
    # to watch live instead.
    echo "[$(date '+%F %T')] Starting ${name}..." >> "${logfile}"
    PYTHONUNBUFFERED=1 "${VENV_PY}" -u "${script}" >> "${logfile}" 2>&1
    echo "[$(date '+%F %T')] ${name} exited, restarting in 2s..." >> "${logfile}"
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
