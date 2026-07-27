#!/usr/bin/env bash
# Reports how many images are sitting in the disk queues, asks for confirmation,
# and only deletes (with progress) if you say yes.
set -uo pipefail

# Matches config.py's QUEUE_ROOT -- kept as a plain path here since this is a
# standalone bash maintenance script, not part of the Python pipeline.
# Overridable via env for testing against a scratch directory instead of the
# real queues.
QUEUE_ROOT="${QUEUE_ROOT:-/mnt/sdb1/jellyscope_queues}"

report_queue_state() {
  for q in que_fullframes que_crops; do
    for d in incoming processing failed; do
      n=$(ls "$QUEUE_ROOT/$q/$d" 2>/dev/null | wc -l)
      echo "  $q/$d: $n file(s)"
    done
  done
}

# que_fullframes items are .tiff, que_crops items are .png -- count those (not
# their .json sidecars) so this reports one number per actual frame/crop, not
# a double-counted image+sidecar total.
n_fullframes=$(find "$QUEUE_ROOT/que_fullframes" -mindepth 2 -name '*.tiff' ! -name '.tmp_*' 2>/dev/null | wc -l)
n_crops=$(find "$QUEUE_ROOT/que_crops" -mindepth 2 -name '*.png' ! -name '.tmp_*' 2>/dev/null | wc -l)
n_images=$((n_fullframes + n_crops))

echo "Queue contents:"
report_queue_state
echo
echo "Total images queued: $n_fullframes full frame(s) + $n_crops crop(s) = $n_images"

# Raw file count (images + .json sidecars + any stray .tmp_* from an interrupted
# atomic write) -- what actually gets deleted below, so the progress bar's
# denominator matches reality even though the summary above talks in "images".
total_files=$(find "$QUEUE_ROOT/que_fullframes" "$QUEUE_ROOT/que_crops" -mindepth 2 -type f 2>/dev/null | wc -l)

if [ "$total_files" -eq 0 ]; then
  echo "Nothing to delete."
  exit 0
fi

read -r -p "Delete all $n_images images ($total_files file(s) total) from both queues? [y/N] " reply
case "$reply" in
  [yY]|[yY][eE][sS]) ;;
  *) echo "Aborted, nothing deleted."; exit 0 ;;
esac

# Deletes files one at a time (not `find -delete`) so progress can be reported --
# subdirectories (incoming/processing/failed) are left in place, not removed,
# so the pipeline doesn't need ensure_queue_dirs() to recreate them before its
# next write.
deleted=0
while IFS= read -r -d '' f; do
  rm -f "$f"
  deleted=$((deleted + 1))
  if (( deleted % 50 == 0 || deleted == total_files )); then
    printf "\rDeleted %d/%d file(s)..." "$deleted" "$total_files"
  fi
done < <(find "$QUEUE_ROOT/que_fullframes" "$QUEUE_ROOT/que_crops" -mindepth 2 -type f -print0)
printf "\rDeleted %d/%d file(s).        \n" "$deleted" "$total_files"

echo
echo "Done. Queue state now:"
report_queue_state
