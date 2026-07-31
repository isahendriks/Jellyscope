#!/usr/bin/env bash
# Prepares a newly-uploaded AE + scorer checkpoint pair (from server-lab's training
# script) for use by Monitor/supervisor.sh's production loop -- everything from
# "checkpoints exist under models/" to "analyse.py is running the new INT8 engines"
# in one command:
#
#   1. Locates models/<name>_AE_model*.pth and models/<name>_scorer_*.pth (binary
#      or mahalanobis, whichever was uploaded -- auto-detected downstream the same
#      way analyse.py already does, see models/segmentation.py's detect_scorer_type()).
#   2. Points Monitor/config.py's SEGMENTATION_AE_MODEL_PATH/SEGMENTATION_SCORER_MODEL_PATH
#      at them, and clears SEGMENTATION_SCORER_THRESHOLD_OVERRIDE back to None so the
#      new checkpoint's own trained threshold is used, not a stale hand-tuned value
#      left over from debugging a previous one.
#   3. Deletes stale seg_encoder/seg_decoder/seg_scorer .onnx and .engine files --
#      export_onnx.py/build_trt_int8.py both skip re-building anything whose output
#      file already exists, which is exactly how a previous AE swap silently kept
#      running on old INT8 weights for hours before anyone noticed (see git history/
#      that incident's postmortem) -- so nothing here is allowed to just get skipped.
#   4. Re-exports ONNX, re-collects INT8 calibration data (samples evenly from
#      <name>/Test/obs/ and .../no_obs/, not just whichever the AE happens to like),
#      and rebuilds the INT8 engines.
#   5. Kills the running analyse.py so Monitor/supervisor.sh's restart-loop relaunches
#      it with the new config/engines -- Monitor/config.py is tracked in git, so
#      `git diff Jetson_monitoring/Monitor/config.py` is your undo button if a run
#      goes wrong.
#
# Usage: ./update_segmentation_model.sh <name> [--with-fp16] [--max-images N] [--calibration-folder DIR]
#   <name>  matches whatever monitoring_effort train_DNN.py was run with, e.g. Kristineberg_260729
#           (case-insensitive -- Kristineberg_260729 and kristineberg_260729 both match)
#
# Calibration images default to /mnt/sda1/TrainingData/Binary_classifier/<name>/Test/
# (both obs/ and no_obs/ must exist there) -- override with --calibration-folder.
#
# set -e (not just -uo pipefail, unlike the *_supervisor.sh restart-loop scripts) is
# deliberate here: this is a one-shot linear pipeline, not a loop that should keep
# going after a stage fails -- if export_onnx.py chokes on a grid_size mismatch,
# collect_calibration_data.py and build_trt_int8.py must NOT run against whatever
# half-broken state that left behind.
set -euo pipefail
shopt -s nullglob
shopt -s nocaseglob  # so `kristineberg_260729` matches `Kristineberg_260729_...` files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="${SCRIPT_DIR}/.venv-jellyscope_on_jetson/bin/python"
MODELS_DIR="${SCRIPT_DIR}/models"
CONFIG_PY="${SCRIPT_DIR}/Monitor/config.py"
ENGINE_DIR="${SCRIPT_DIR}/trt/engines"
ONNX_DIR="${SCRIPT_DIR}/trt/onnx"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <name> [--with-fp16] [--max-images N] [--calibration-folder DIR]" >&2
  echo "  e.g.: $0 Kristineberg_260729" >&2
  exit 1
fi
NAME="$1"
shift

WITH_FP16=false
MAX_IMAGES=60
CAL_FOLDER="/mnt/sda1/TrainingData/Binary_classifier/${NAME}/Test"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-fp16) WITH_FP16=true; shift ;;
    --max-images) MAX_IMAGES="$2"; shift 2 ;;
    --calibration-folder) CAL_FOLDER="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

echo "== Preparing '${NAME}' for production =="
echo

# --- 1. Locate the new checkpoints -------------------------------------------------
ae_matches=("${MODELS_DIR}/${NAME}_AE_model"*.pth)
if [[ "${#ae_matches[@]}" -eq 0 ]]; then
  echo "ERROR: no AE checkpoint matching ${MODELS_DIR}/${NAME}_AE_model*.pth" >&2
  exit 1
elif [[ "${#ae_matches[@]}" -gt 1 ]]; then
  echo "ERROR: multiple AE checkpoints match '${NAME}_AE_model*.pth' -- ambiguous:" >&2
  printf '  %s\n' "${ae_matches[@]}" >&2
  echo "Delete/rename the one(s) you don't want and re-run." >&2
  exit 1
fi
ae_path="${ae_matches[0]}"
echo "AE checkpoint:     $(basename "${ae_path}")"

scorer_matches=("${MODELS_DIR}/${NAME}_scorer_"*.pth)
if [[ "${#scorer_matches[@]}" -eq 0 ]]; then
  echo "ERROR: no scorer checkpoint matching ${MODELS_DIR}/${NAME}_scorer_*.pth" >&2
  exit 1
elif [[ "${#scorer_matches[@]}" -gt 1 ]]; then
  echo "ERROR: multiple scorer checkpoints match '${NAME}_scorer_*.pth' -- ambiguous "\
"(maybe both a binary and a mahalanobis run got uploaded for this name):" >&2
  printf '  %s\n' "${scorer_matches[@]}" >&2
  echo "Delete/rename the one(s) you don't want and re-run." >&2
  exit 1
fi
scorer_path="${scorer_matches[0]}"
echo "Scorer checkpoint: $(basename "${scorer_path}")"
echo

# --- 2. Calibration data must actually be there -------------------------------------
if [[ ! -d "${CAL_FOLDER}/obs" && ! -d "${CAL_FOLDER}/no_obs" ]]; then
  echo "ERROR: neither ${CAL_FOLDER}/obs nor ${CAL_FOLDER}/no_obs exists." >&2
  echo "Copy the calibration images there first (or pass --calibration-folder)." >&2
  exit 1
fi
n_obs=$(find "${CAL_FOLDER}/obs" -iname '*.png' 2>/dev/null | wc -l)
n_no_obs=$(find "${CAL_FOLDER}/no_obs" -iname '*.png' 2>/dev/null | wc -l)
echo "Calibration images under ${CAL_FOLDER}: ${n_obs} obs, ${n_no_obs} no_obs"
if [[ "${n_obs}" -eq 0 && "${n_no_obs}" -eq 0 ]]; then
  echo "ERROR: no .png images found under ${CAL_FOLDER}/{obs,no_obs}" >&2
  exit 1
fi
echo

# --- 3. Point config.py at the new checkpoints, clear any manual threshold override -
sed -i "s|^SEGMENTATION_AE_MODEL_PATH = .*|SEGMENTATION_AE_MODEL_PATH = PIPELINE_DIR / \"models\" / \"$(basename "${ae_path}")\"|" "${CONFIG_PY}"
sed -i "s|^SEGMENTATION_SCORER_MODEL_PATH = .*|SEGMENTATION_SCORER_MODEL_PATH = PIPELINE_DIR / \"models\" / \"$(basename "${scorer_path}")\"|" "${CONFIG_PY}"
sed -i "s|^SEGMENTATION_SCORER_THRESHOLD_OVERRIDE = .*|SEGMENTATION_SCORER_THRESHOLD_OVERRIDE = None  # reset by update_segmentation_model.sh for '${NAME}' -- re-add manually if you want to override its trained threshold|" "${CONFIG_PY}"
echo "Updated Monitor/config.py:"
grep -n "^SEGMENTATION_AE_MODEL_PATH\|^SEGMENTATION_SCORER_MODEL_PATH\|^SEGMENTATION_SCORER_THRESHOLD_OVERRIDE" "${CONFIG_PY}" | sed 's/^/  /'
echo

# --- 4. Wipe stale SEGMENT onnx/engine files so nothing gets silently skipped ------
rm -f "${ENGINE_DIR}"/seg_encoder_int8.engine "${ENGINE_DIR}"/seg_decoder_int8.engine "${ENGINE_DIR}"/seg_scorer_int8.engine
rm -f "${ONNX_DIR}"/seg_encoder.onnx "${ONNX_DIR}"/seg_encoder.onnx.data
rm -f "${ONNX_DIR}"/seg_decoder.onnx "${ONNX_DIR}"/seg_decoder.onnx.data
rm -f "${ONNX_DIR}"/seg_scorer.onnx "${ONNX_DIR}"/seg_scorer.onnx.data
if [[ "${WITH_FP16}" != true ]]; then
  rm -f "${ENGINE_DIR}"/seg_encoder_fp16.engine "${ENGINE_DIR}"/seg_decoder_fp16.engine "${ENGINE_DIR}"/seg_scorer_fp16.engine
fi
echo "Cleared stale SEGMENT onnx/engine files (ViT's are untouched -- this only ever touches AE/scorer)."
echo

# --- 5. Export -> calibrate -> build INT8 -------------------------------------------
echo "-- export_onnx.py --"
"${VENV_PY}" "${SCRIPT_DIR}/engines/export_onnx.py"
echo

echo "-- collect_calibration_data.py --"
"${VENV_PY}" "${SCRIPT_DIR}/engines/collect_calibration_data.py" "${CAL_FOLDER}" --max-images "${MAX_IMAGES}"
echo

echo "-- build_trt_int8.py --"
build_args=()
if [[ "${WITH_FP16}" != true ]]; then
  build_args+=(--skip-fp16)
fi
"${VENV_PY}" "${SCRIPT_DIR}/engines/build_trt_int8.py" "${build_args[@]}"
echo

# --- 6. Restart analyse.py so Monitor/supervisor.sh's loop picks everything up ------
echo "-- restarting analyse.py --"
if pkill -f "Monitor/analyse.py"; then
  echo "  killed analyse.py -- Monitor/supervisor.sh will relaunch it within ~2s"
else
  echo "  analyse.py wasn't running (nothing to restart -- start Monitor/supervisor.sh when ready)"
fi

echo
echo "== Done. '${NAME}' is now the active SEGMENT model. =="
echo "Confirm with: grep -A1 \"SEGMENT scorer\" Monitor/logs/analyse.log | tail -5"
