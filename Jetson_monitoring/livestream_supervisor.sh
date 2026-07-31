#!/usr/bin/env bash
# Runs livestream.py + metadata.py, each in its own restart-loop, same pattern as
# Monitor/supervisor.sh's four-stage version -- for standalone livestream.py sessions
# (dev/live-view capture, not the full record/analyse/send pipeline) that still want
# metadata.py running alongside so the live-stream page's ENVIRONMENTAL/LEAK/DEVICE
# panel (see livestream.py's _read_live_metadata()) has real data instead of N/A.
# Either stage crashing and restarting doesn't affect the other. Ctrl-C stops both.
#
# Don't run this at the same time as Monitor/supervisor.sh -- livestream.py and
# record.py both open the same camera device, and only one process can hold it.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="${SCRIPT_DIR}/.venv-jellyscope_on_jetson/bin/python"
LOG_DIR="${SCRIPT_DIR}/Monitor/logs"
mkdir -p "${LOG_DIR}"

run_loop() {
  # NOT `local name="$1" logfile="...${name}..."` on one line -- bash expands every
  # RHS in a compound `local` statement before any of them actually become local
  # variables, so `${name}` there gets read while `name` doesn't exist yet, which
  # `set -u` (correctly) flags as unbound. Sequential statements avoid that.
  local name="$1" script="$2"
  local logfile="${LOG_DIR}/${name}.log"
  while true; do
    # Straight file redirection, not `| tee -a` to this terminal -- see
    # Monitor/supervisor.sh's run_loop comment for why: piping through tee ties the
    # process's liveness to this terminal/SSH session, and a half-dead connection can
    # block tee's write() forever, backing up into the python process's own print()
    # and hanging it silently -- the restart loop then never fires either. Plain `>>`
    # to a file has no such reader to block on. Use `tail -f logs/<name>.log` instead.
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

# A single direct echo to this terminal, not through any per-stage pipe -- safe (see
# run_loop's comment above) and worth it, since a silent terminal has been mistaken
# for a hang before.
echo "Starting livestream/metadata/tegrastats. Each stage's output goes to its own"
echo "file under ${LOG_DIR}/ -- NOT to this terminal -- so seeing nothing print here"
echo "is normal, not a hang. Watch live progress with:"
echo "  tail -f ${LOG_DIR}/livestream.log"
echo "Ctrl-C here stops all stages."
echo

run_loop "livestream" "${SCRIPT_DIR}/livestream.py"      &
run_loop "metadata"   "${SCRIPT_DIR}/Monitor/metadata.py" &
run_tegrastats_loop                                       &

# `kill $(jobs -p)` only signals the three run_loop/run_tegrastats_loop subshells
# themselves, not the venv python / tegrastats processes they run in the foreground --
# those are separate child PIDs the subshell is merely wait()-ing on, and SIGTERM to
# the subshell doesn't propagate to them. Confirmed by testing: after that kill, the
# child kept running, orphaned, still holding the camera device and appending to its
# log. Since this script isn't run with job control (`set -m`), all of its descendants
# share its own PGID, so signaling the negative PGID reaches every stage's actual
# worker process too -- but that also re-signals this script's own PID (it's a member
# of its own group), which would re-enter this same trap forever. `trap - INT TERM`
# first drops back to the default disposition so the repeat signal just lets the
# script die instead of recursing (confirmed by testing: without it, "Stopping all
# stages..." loops indefinitely).
trap 'echo "Stopping all stages..."; trap - INT TERM; kill -TERM -- -$$ 2>/dev/null' INT TERM

wait
