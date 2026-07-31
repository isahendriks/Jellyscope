#!/usr/bin/env bash
# Pushes new training-data files from this Jetson's local corpus to server-lab's R:
# network drive over SSH/scp, copying only what isn't already there.
#
# rsync isn't an option here: Monitor/transfer.py already found server-lab (a Windows
# box) has no rsync binary, and rsync needs one on *both* ends since it invokes a
# remote `rsync --server` process. This script instead reimplements just the one piece
# of rsync's behavior actually needed -- "skip whatever's already there" -- by listing
# both sides' files once (by relative path) and scp-ing only what's missing remotely,
# batched per destination subdirectory so files sharing a folder go in one scp call
# instead of one round-trip each (same batching idea as transfer.py's scp_upload()).
#
# Usage: ./sync_trainingdata.sh [local_dir] [remote_ssh_target] [remote_windows_path]
#   Defaults:
#     local_dir            /mnt/sda1/TrainingData
#     remote_ssh_target    jellyfish@server-lab
#     remote_windows_path  R:\LU24A1037-Jellyscope\Jellyscope\Training data new
#
# Assumes the remote shell for non-interactive SSH commands is cmd.exe (Win32-OpenSSH's
# default, and what transfer.py's own mkdir already relies on) -- specifically for the
# `dir /s /b /a-d` listing below. Not yet run against the real server-lab box; if its
# SSH server is configured to use PowerShell as the default shell instead, that one
# command (and only that one) would need adjusting to Get-ChildItem syntax. Watch the
# first real run's "already present remotely" count -- if it's suspiciously 0 on a
# second run that should be a no-op, that command is the first thing to check.
#
# set -uo pipefail (not -e): one failed mkdir/scp for a single subdirectory group
# shouldn't abort the whole sync -- the summary at the end reports what did/didn't
# make it, same "keep going, report failures" spirit as *_supervisor.sh's restart loops.
set -uo pipefail

LOCAL_DIR="${1:-/mnt/sda1/TrainingData}"
REMOTE_HOST="${2:-jellyfish@server-lab}"
REMOTE_WIN_PATH="${3:-R:\\LU24A1037-Jellyscope\\Jellyscope\\Training data new}"

# scp's destination argument uses forward slashes -- a Windows path's drive-letter colon
# (e.g. "R:\...") looks to scp's legacy host:path parser like ANOTHER "host:" prefix if
# given backslashes/colons as-is; forward slashes sidestep that ambiguity entirely, and
# Windows accepts either separator for the actual file operation. The mkdir/dir commands
# below still use real backslashes since those are parsed by cmd.exe, not scp.
REMOTE_FWD_PATH="${REMOTE_WIN_PATH//\\//}"

SSH_CONNECT_TIMEOUT_S=10
LIST_TIMEOUT_S=300  # a full-tree `dir /s /b` can take a while once the archive is large

CONTROL_PATH="/tmp/jellyscope-ssh-ctrl-${REMOTE_HOST}"
# Same ControlMaster/ControlPath/ControlPersist reuse as Monitor/transfer.py -- every
# ssh/scp call below reuses one real connection instead of re-handshaking each time.
SSH_OPTS=(-o BatchMode=yes -o "ConnectTimeout=${SSH_CONNECT_TIMEOUT_S}" -o StrictHostKeyChecking=accept-new
          -o ControlMaster=auto -o "ControlPath=${CONTROL_PATH}" -o ControlPersist=600)

echo "== sync_trainingdata: ${LOCAL_DIR} -> ${REMOTE_HOST}:${REMOTE_WIN_PATH} =="
echo

if [[ ! -d "${LOCAL_DIR}" ]]; then
  echo "ERROR: local directory not found: ${LOCAL_DIR}" >&2
  exit 1
fi

echo "Checking SSH connectivity to ${REMOTE_HOST}..."
if ! ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "whoami" > /dev/null 2>&1; then
  echo "ERROR: SSH to ${REMOTE_HOST} failed. Confirm key auth works: ssh -o BatchMode=yes ${REMOTE_HOST} whoami" >&2
  exit 1
fi

echo "Ensuring destination exists: ${REMOTE_WIN_PATH}"
# Best-effort -- cmd.exe's mkdir isn't idempotent like `mkdir -p`, "already exists" is
# the expected/common case here, same convention as Monitor/transfer.py's ensure_remote_dir.
ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "mkdir \"${REMOTE_WIN_PATH}\"" > /dev/null 2>&1

echo "Listing local files under ${LOCAL_DIR}..."
mapfile -t local_files < <(cd "${LOCAL_DIR}" && find . -type f | sed 's|^\./||' | sort)
echo "  ${#local_files[@]} local file(s)"
echo

if [[ "${#local_files[@]}" -eq 0 ]]; then
  echo "Nothing to sync."
  exit 0
fi

echo "Listing files already present on ${REMOTE_HOST} (this can take a while for a large archive)..."
remote_listing="$(ssh "${SSH_OPTS[@]}" -o "ConnectTimeout=${LIST_TIMEOUT_S}" "${REMOTE_HOST}" \
  "dir /s /b /a-d \"${REMOTE_WIN_PATH}\"" 2>/dev/null || true)"

declare -A remote_set=()
prefix_warnings=0
if [[ -n "${remote_listing}" ]]; then
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    line="${line%$'\r'}"  # strip a trailing \r if the remote line-endings are CRLF
    rel="${line#"${REMOTE_WIN_PATH}"\\}"
    if [[ "${rel}" == "${line}" ]]; then
      # Prefix didn't match (unexpected path normalization on the remote side) -- don't
      # guess. Leaving this line out of remote_set means the corresponding local file (if
      # any) gets copied again rather than silently skipped -- a harmless redundant copy
      # is far better than silently missing a genuinely-new file.
      prefix_warnings=$((prefix_warnings + 1))
      continue
    fi
    rel="${rel//\\//}"
    remote_set["${rel}"]=1
  done <<< "${remote_listing}"
fi
echo "  ${#remote_set[@]} file(s) already present remotely"
if [[ "${prefix_warnings}" -gt 0 ]]; then
  echo "  NOTE: ${prefix_warnings} remote line(s) didn't match the expected path prefix and were" \
       "ignored (treated as not-yet-present, so nothing gets silently skipped -- see this" \
       "script's header comment about the cmd.exe assumption)."
fi
echo

missing=()
for f in "${local_files[@]}"; do
  if [[ -z "${remote_set[${f}]+x}" ]]; then
    missing+=("${f}")
  fi
done
echo "  ${#missing[@]} file(s) missing remotely"
echo

if [[ "${#missing[@]}" -eq 0 ]]; then
  echo "Already up to date, nothing to copy."
  exit 0
fi

# Group missing files by destination subdirectory so files sharing a folder go in one
# scp call, not one round-trip per file.
declare -A groups=()  # relative subdir -> newline-separated relative file paths
for f in "${missing[@]}"; do
  subdir="$(dirname "${f}")"
  groups["${subdir}"]+="${f}"$'\n'
done

copied=0
failed=0
total="${#missing[@]}"
for subdir in "${!groups[@]}"; do
  if [[ "${subdir}" == "." ]]; then
    remote_subdir_win="${REMOTE_WIN_PATH}"
    remote_subdir_fwd="${REMOTE_FWD_PATH}"
  else
    remote_subdir_win="${REMOTE_WIN_PATH}\\${subdir//\//\\}"
    remote_subdir_fwd="${REMOTE_FWD_PATH}/${subdir}"
  fi
  ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "mkdir \"${remote_subdir_win}\"" > /dev/null 2>&1

  files_in_group=()
  while IFS= read -r rel; do
    [[ -z "${rel}" ]] && continue
    files_in_group+=("${LOCAL_DIR}/${rel}")
  done <<< "${groups[${subdir}]}"

  if scp "${SSH_OPTS[@]}" "${files_in_group[@]}" "${REMOTE_HOST}:${remote_subdir_fwd}/" > /dev/null 2>&1; then
    copied=$((copied + ${#files_in_group[@]}))
  else
    failed=$((failed + ${#files_in_group[@]}))
    echo
    echo "  WARNING: scp failed for '${subdir}' (${#files_in_group[@]} file(s))" >&2
  fi
  printf "\rCopied %d/%d, failed %d..." "${copied}" "${total}" "${failed}"
done
printf "\rCopied %d/%d, failed %d.        \n" "${copied}" "${total}" "${failed}"
echo

if [[ "${failed}" -eq 0 ]]; then
  echo "Done. ${copied} file(s) copied to ${REMOTE_HOST}:${REMOTE_WIN_PATH}"
else
  echo "Done with ${failed} failure(s) -- re-run to retry (already-copied files will be skipped)."
fi
