#!/bin/bash

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# get path variables from paths.env
source "$SCRIPT_DIR/paths.env"

# VIDEO_PATH="$Result_OUTPUT_ROOT/unity_curve-deepspeed_o2-lr-1e-5-step-30000-worldsize-4-length-25-H-512/eval_unity_curve"
# VIDEO_PATH="$Result_OUTPUT_ROOT/unity_curve-deepspeed_o2-lr-1e-5-step-30000-worldsize-4-length-25-H-576/eval_unity_curve"

VIDEO_PATH="$Result_OUTPUT_ROOT/d6250ea37f38341f49dfe1009402e3684c2efc93/eval_unity_curve"


NUM_VIDEOS=200 # 200

# args for metric calculation
FVD_FULL_ONLY=0
READ_WORKERS=32
SSIM_BATCH_SIZE=32
PSNR_BATCH_SIZE=64
export LATENT_BATCH_SIZE=32


FVD_ARGS=""
if [ "$FVD_FULL_ONLY" = "1" ]; then
    FVD_ARGS="--fvd_full_only"
fi

# Evaluate all frames from existing segment folders
# RESULT_PATH="eval_score_all.json"
RESULT_PATH="eval_score_all_multi_fvd.json"

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

# # Evaluate specific segment
# SEGMENT_ID=2  # Change this to evaluate different segments (0, 1, 2)
# RESULT_PATH="eval_score_sgemnet_${SEGMENT_ID}.json"

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
