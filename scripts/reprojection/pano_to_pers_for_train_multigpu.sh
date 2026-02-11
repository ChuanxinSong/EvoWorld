#!/bin/bash

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Base directory containing episode folders.
# BASE_DIR="/data1/songcx/dataset/evo_world/unity_curve/train"
BASE_DIR="/data2/songcx/dataset/evoworld/unity_curve/train"

MAX_JOBS=8  # <--- 这里设置并发进程数，建议设置为你 CPU 核心数的 80% 左右

echo "Starting processing with $MAX_JOBS parallel jobs..."

# 定义计数器
count=0

for EPISODE_PATH in "$BASE_DIR"/*; do
    
    # 确保是目录
    if [[ -d "$EPISODE_PATH" ]]; then
        
        # (后台执行逻辑)
        (
            echo "Processing: $EPISODE_PATH"
            
            SOURCE_DIR="$EPISODE_PATH/panorama"
            TARGET_DIR="$EPISODE_PATH/perspective_look_at_center"
            CAMERA_FILE="$EPISODE_PATH/camera_poses.txt"
            OUTPUT_CAMERA_FILE="$EPISODE_PATH/camera_poses_look_at_center.txt"
            
            # 运行 Python 脚本
            python -m evoworld.reprojection.pano_to_pers \
                --data_folder "$SOURCE_DIR" \
                --output_folder "$TARGET_DIR" \
                --camera_file "$CAMERA_FILE" \
                --output_camera_file "$OUTPUT_CAMERA_FILE"
                
            echo "Finished: $EPISODE_PATH"
        ) &  # <--- 关键点：加 & 符号放入后台运行

        # 进程控制逻辑
        ((count++))
        if (( count >= MAX_JOBS )); then
            wait -n  # 等待任意一个后台任务结束
            ((count--))
        fi
        
    fi
done

wait # 等待最后剩余的所有任务结束
echo "All done!"
