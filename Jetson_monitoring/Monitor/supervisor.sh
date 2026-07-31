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

run_tegrastats_loop() {
  # Same restart-loop shape as run_loop, but wraps the tegrastats binary instead of a
  # venv script -- logs CPU/GPU/RAM/thermal/power every 5s so a system-wide freeze
  # (like the 2026-07-29 watchdog reset -- see leak_alert.py's git history / postmortem)
  # leaves telemetry up to the last sample instead of nothing at all.
  local logfile="${LOG_DIR}/tegrastats.log"
  while true; do
    echo "[$(date '+%F %T')] Starting tegrastats..." >> "${logfile}"
    tegrastats --interval 5000 >> "${logfile}" 2>&1
    echo "[$(date '+%F %T')] tegrastats exited, restarting in 2s..." >> "${logfile}"
    sleep 2
  done
}

cd "${SCRIPT_DIR}"

# A single direct echo to this terminal, not through any per-stage pipe -- safe
# (nothing to deadlock on, see run_loop's comment above) and worth it, since a
# silent terminal here has twice now been mistaken for a hang when the pipeline
# was actually running fine.
echo "Starting record/analyse/send/metadata/tegrastats. Each stage's output goes to"
echo "its own file under ${LOG_DIR}/ -- NOT to this terminal -- so seeing nothing"
echo "print here is normal, not a hang. Watch live progress with:"
echo "  tail -f ${LOG_DIR}/analyse.log"
echo "Ctrl-C here stops all stages."
echo

run_loop "record"   "${SCRIPT_DIR}/record.py"   &
run_loop "analyse"  "${SCRIPT_DIR}/analyse.py"  &
run_loop "send"     "${SCRIPT_DIR}/send.py"     &
run_loop "metadata" "${SCRIPT_DIR}/metadata.py" &
run_tegrastats_loop                             &

# `kill $(jobs -p)` only signals the run_loop/run_tegrastats_loop subshells
# themselves, not the venv python / tegrastats processes they run in the foreground --
# those are separate child PIDs the subshell is merely wait()-ing on, and SIGTERM to
# the subshell doesn't propagate to them, leaving them orphaned and still holding the
# camera device (confirmed by testing in livestream_supervisor.sh, which had the same
# pattern). Since this script isn't run with job control (`set -m`), all of its
# descendants share its own PGID, so signaling the negative PGID reaches every stage's
# actual worker process too -- but that also re-signals this script's own PID (it's a
# member of its own group), which would re-enter this same trap forever. `trap - INT
# TERM` first drops back to the default disposition so the repeat signal just lets the
# script die instead of recursing.
trap 'echo "Stopping all stages..."; trap - INT TERM; kill -TERM -- -$$ 2>/dev/null' INT TERM

wait
