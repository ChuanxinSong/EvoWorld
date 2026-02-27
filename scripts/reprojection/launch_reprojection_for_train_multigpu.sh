#!/bin/bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
SCRIPT_PATH="scripts/reprojection/reproject_vggt_open3d_for_train.sh"

# 优雅退出：按下 Ctrl+C 时杀掉所有子进程并退出
trap "echo 'Terminating all processes...'; pkill -P $$; exit 1" SIGINT SIGTERM

# ...existing code...
DATA_FOLDER="/data2/songcx/dataset/evoworld/unity_curve/train" # 确保路径正确
CHUNK_NUM=16      # 总分块数

# 根据你的权限配置：0和1卡多跑一些（各3个），3和5卡各跑1个
# 这样总共会有 8 个并行任务
GPUS=(3 4 4 5 5 7) 
MAX_JOBS=${#GPUS[@]}        

echo "Starting parallel reprojection with $MAX_JOBS jobs on GPUs: ${GPUS[*]}..."

# 2. 并行执行逻辑
count=0
for CHUNK_ID in $(seq 0 $((CHUNK_NUM-1))); do
    # 计算当前任务应该分配给哪个 GPU
    GPU_ID=${GPUS[$((count % MAX_JOBS))]}
    
    echo "Launching Chunk $CHUNK_ID on GPU $GPU_ID..."
    
    # 使用 CUDA_VISIBLE_DEVICES 隔离显卡并后台运行
    CUDA_VISIBLE_DEVICES=$GPU_ID bash "$SCRIPT_PATH" "$DATA_FOLDER" "$CHUNK_ID" "$CHUNK_NUM" &
    
    ((count++))
    
    # 达到最大并行数时，等待任意一个进程结束再继续
    if (( count >= MAX_JOBS )); then
        wait -n
    fi
done

# 等待剩余所有后台任务结束
wait
echo "All $CHUNK_NUM chunks completed across all GPUs."
