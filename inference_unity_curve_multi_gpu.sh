#!/bin/bash
set -e
set -o pipefail

# Unified Loop Consistency Pipeline Runner
# This script runs the unified pipeline that combines generation and reconstruction

# link to correct cudnn and cuda
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# source paths.env, get BASE_FOLDER and OUTPUT_ROOT
source "$SCRIPT_DIR/paths.env"

# CKPT=MODELS/evoworld_curve_unity
# CKPT=/data1/songcx/huggingface_cache/hub/models--CometsFeiyu--Evoworld_Unity_Curve_Path/snapshots/d6250ea37f38341f49dfe1009402e3684c2efc93/

CKPT=/home/user/songcx/code/EvoWorld/evo_checkpoints/unity_curve-deepspeed_o2-lr-1e-5-step-30000-worldsize-4-length-25-H-576
# CKPT=/home/user/songcx/code/EvoWorld/evo_checkpoints/unity_curve-deepspeed_o2-lr-1e-5-step-30000-worldsize-4-length-25-H-512

OUTPUT_ROOT="$Result_OUTPUT_ROOT"
SAVE_DIR="$OUTPUT_ROOT/$(basename "$CKPT")/eval_unity_curve"
START_IDX=0 # starting episode index for data loading, change it if some episodes have been processed and saved in the save dir.
NUM_DATA_PER_GPU=100 # Adjust this based on your NUM_GPUS: total data = NUM_GPUS * NUM_DATA_PER_GPU
NUM_SEGMENTS=3
CURVE_PATH=true
SKIP_COMPLETED=true
WIDTH=1024
HEIGHT=576 # 576 | 512
GPU_IDS="${GPU_IDS:-5,4}" # comma-separated GPU ids, e.g. GPU_IDS=2,3
LOG_TO_CONSOLE="${LOG_TO_CONSOLE:-true}"

mkdir -p "$SAVE_DIR"

IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ID_ARRAY[@]}

echo "Using GPU_IDS=$GPU_IDS (NUM_GPUS=$NUM_GPUS)"

echo "Running unified loop consistency pipeline..."
echo "Checkpoint: $CKPT"
echo "Base folder: $BASE_FOLDER"
echo "Save dir: $SAVE_DIR"
echo "Number of segments: $NUM_SEGMENTS"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "Skip completed: $SKIP_COMPLETED"
echo "Log to console: $LOG_TO_CONSOLE"

PIDS=()

cleanup() {
    echo "Stopping ${#PIDS[@]} launched GPU process groups..."
    trap - INT TERM
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    exit 130
}

trap cleanup INT TERM

for GPU_INDEX in "${!GPU_ID_ARRAY[@]}"; do
    GPU_ID="${GPU_ID_ARRAY[$GPU_INDEX]}"
    # compute start idx for this GPU
    PROC_START_IDX=$(( START_IDX + GPU_INDEX * NUM_DATA_PER_GPU ))


    # build per-process command
    CMD_GPU="python unified_loop_consistency.py \
        --unet_path $CKPT \
        --svd_path $CKPT \
        --base_folder $BASE_FOLDER \
        --save_dir $SAVE_DIR \
        --num_data $NUM_DATA_PER_GPU \
        --start_idx $PROC_START_IDX \
        --num_segments $NUM_SEGMENTS \
        --num_frames 25 \
        --clip_start_frame 1 \
        --width $WIDTH \
        --height $HEIGHT \
        --save_frames"

    if [ "$CURVE_PATH" = true ]; then
        CMD_GPU="$CMD_GPU --curve_path"
    fi

    if [ "$SKIP_COMPLETED" = true ]; then
        CMD_GPU="$CMD_GPU --skip_completed"
    fi

    echo "Launching GPU $GPU_ID -> start_idx=$PROC_START_IDX, save_dir=$SAVE_DIR"
    echo "Command: CUDA_VISIBLE_DEVICES=$GPU_ID $CMD_GPU"

    # Run the process bound to this GPU in background and capture its PID.
    # Prefix each line when streaming to the shared console so interleaved logs remain readable.
    if [ "$LOG_TO_CONSOLE" = true ]; then
        RUN_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID $CMD_GPU 2>&1 | sed -u 's/^/[GPU $GPU_ID] /' | tee '$SAVE_DIR/run_gpu${GPU_ID}.log'"
    else
        RUN_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID $CMD_GPU > '$SAVE_DIR/run_gpu${GPU_ID}.log' 2>&1"
    fi
    setsid bash -c "$RUN_CMD" &
    PIDS+=("$!")
done

echo "Launched ${#PIDS[@]} processes, waiting for them to finish..."

# wait for all background processes
for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "Process $pid exited with non-zero status"
done

echo "Unified pipeline completed for all GPUs!"
