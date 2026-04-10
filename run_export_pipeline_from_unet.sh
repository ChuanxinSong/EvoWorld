#!/bin/bash

# Export a full SVD pipeline from a trained UNet checkpoint.
#
# Usage:
#   bash run_export_pipeline_from_unet.sh <UNET_PATH> [SVD_PATH] [OUTPUT_DIR]
#
# Example:
#   bash run_export_pipeline_from_unet.sh \
#     evo_checkpoints/unity_curve-OnlyPlucker-deepspeed_o2-lr-1e-5-step-30000-worldsize-4-length-25/checkpoint-10000

set -euo pipefail

# Optional: link to correct cudnn and cuda
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

SVD_PATH_DEFAULT="stabilityai/stable-video-diffusion-img2vid-xt-1-1"

UNET_PATH_DEFAULT="evo_checkpoints/unity_curve-OnlyPlucker-deepspeed_o2-lr-1e-5-step-30000-worldsize-4-length-25-only-position/checkpoint-10000"


UNET_PATH="${1:-$UNET_PATH_DEFAULT}"
SVD_PATH="${2:-$SVD_PATH_DEFAULT}"
OUTPUT_DIR="${3:-${UNET_PATH}_pipeline}"

# Set to 1 if the base SVD model is already cached locally.
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

echo "Exporting pipeline from UNet checkpoint..."
echo "UNET_PATH: $UNET_PATH"
echo "SVD_PATH: $SVD_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "LOCAL_FILES_ONLY: $LOCAL_FILES_ONLY"

CMD=(
    python scripts/export_pipeline_from_unet.py
    --unet_path "$UNET_PATH"
    --svd_path "$SVD_PATH"
    --output_dir "$OUTPUT_DIR"
)

if [ "$LOCAL_FILES_ONLY" = "1" ]; then
    CMD+=(--local_files_only)
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Pipeline export completed."
