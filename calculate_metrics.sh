#!/bin/bash

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# get path variables from paths.env
source "$SCRIPT_DIR/paths.env"

# CKPT=MODELS/evoworld_curve_unity
CKPT=/home/user/songcx/code/EvoWorld/evo_checkpoints/unity_curve-deepspeed_o2-lr-1e-5-step-30000-worldsize-4-length-25-H-512

# OUTPUT_ROOT=output
OUTPUT_ROOT="$Result_OUTPUT_ROOT"
VIDEO_PATH=$OUTPUT_ROOT/$(basename "$CKPT")/eval_unity_curve

RESULT_PATH="eval_score_all.json"
# RESULT_PATH="eval_score.json"
NUM_VIDEOS=180 # 200

# args for metric calculation
READ_WORKERS=32
SSIM_BATCH_SIZE=32
PSNR_BATCH_SIZE=64
export LATENT_BATCH_SIZE=32
FVD_FULL_ONLY=1


FVD_ARGS=""
if [ "$FVD_FULL_ONLY" = "1" ]; then
    FVD_ARGS="--fvd_full_only"
fi

# Evaluate all frames from existing segment folders
python -m evoworld.metrics.calculate_all_metrics \
    --data_path "$VIDEO_PATH" \
    --gt_subdir "predictions_gt_0,predictions_gt_1,predictions_gt_2" \
    --gen_subdir "predictions_0,predictions_1,predictions_2" \
    --result_file "$RESULT_PATH" \
    --test_length 73 \
    --num_videos "$NUM_VIDEOS" \
    --read_workers "$READ_WORKERS" \
    --ssim_batch_size "$SSIM_BATCH_SIZE" \
    --psnr_batch_size "$PSNR_BATCH_SIZE" \
    $FVD_ARGS

# Evaluate specific segment
# SEGMENT_ID=2  # Change this to evaluate different segments (0, 1, 2)
# python -m evoworld.metrics.calculate_all_metrics \
#     --data_path "$VIDEO_PATH" \
#     --gt_subdir "predictions_gt_$SEGMENT_ID" \
#     --gen_subdir "predictions_$SEGMENT_ID" \
#     --result_file "$RESULT_PATH" \
#     --test_length 25 \
#     --num_videos "$NUM_VIDEOS" \
#     --read_workers "$READ_WORKERS" \
#     --ssim_batch_size "$SSIM_BATCH_SIZE" \
#     --psnr_batch_size "$PSNR_BATCH_SIZE" \
#     $FVD_ARGS
