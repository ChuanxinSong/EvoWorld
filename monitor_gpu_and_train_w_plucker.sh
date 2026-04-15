#!/bin/bash

# 监控 2 号和 4 号 GPU 的空闲显存，两个卡都 > 45G 时启动训练

TARGET_GPUS="2,4"
MEMORY_THRESHOLD_GB=45
CHECK_INTERVAL=2
TRAIN_SCRIPT="train_w_plucker.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

threshold_mb=$((MEMORY_THRESHOLD_GB * 1024))
declare -a target_gpu_array=()
declare -A gpu_free_memory_map=()
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

parse_target_gpus() {
    IFS=',' read -r -a target_gpu_array <<< "$TARGET_GPUS"

    if [ "${#target_gpu_array[@]}" -eq 0 ]; then
        log_error "TARGET_GPUS 不能为空"
        exit 1
    fi

    local i
    local gpu_id
    for i in "${!target_gpu_array[@]}"; do
        gpu_id=$(echo "${target_gpu_array[$i]}" | xargs)

        if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
            log_error "非法 GPU ID: '${target_gpu_array[$i]}'"
            exit 1
        fi

        target_gpu_array[$i]="$gpu_id"
    done
}

check_dependencies() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_error "nvidia-smi 未找到，请确认 NVIDIA 驱动已安装"
        exit 1
    fi

    if [ ! -f "$TRAIN_SCRIPT" ]; then
        log_error "训练脚本 '$TRAIN_SCRIPT' 未找到"
        exit 1
    fi

    parse_target_gpus

    local gpu_count
    gpu_count=$(nvidia-smi --list-gpus | wc -l)

    local gpu_id
    for gpu_id in "${target_gpu_array[@]}"; do
        if [ "$gpu_id" -ge "$gpu_count" ]; then
            log_error "GPU $gpu_id 不存在，当前机器共有 $gpu_count 张 GPU"
            exit 1
        fi
    done
}

query_all_gpu_free_memory_mb() {
    nvidia-smi \
        --query-gpu=index,memory.free \
        --format=csv,noheader,nounits 2>/dev/null
}

collect_target_gpu_free_memory() {
    gpu_free_memory_map=()

    local query_output
    query_output=$(query_all_gpu_free_memory_mb)

    if [ -z "$query_output" ]; then
        return 1
    fi

    local gpu_id
    local free_memory_mb
    while IFS=',' read -r gpu_id free_memory_mb; do
        gpu_id=$(echo "$gpu_id" | xargs)
        free_memory_mb=$(echo "$free_memory_mb" | xargs)

        if [[ "$gpu_id" =~ ^[0-9]+$ ]] && [[ "$free_memory_mb" =~ ^[0-9]+$ ]]; then
            gpu_free_memory_map["$gpu_id"]="$free_memory_mb"
        fi
    done <<< "$query_output"

    local target_gpu
    for target_gpu in "${target_gpu_array[@]}"; do
        if [ -z "${gpu_free_memory_map[$target_gpu]+x}" ]; then
            return 1
        fi
    done

    return 0
}

format_mb_to_gb() {
    awk "BEGIN { printf \"%.2f\", $1 / 1024 }"
}

run_training() {
    log_success "GPU ${TARGET_GPUS} 空闲显存都大于 ${MEMORY_THRESHOLD_GB}GB，开始执行 $TRAIN_SCRIPT"
    bash "$TRAIN_SCRIPT"
    return $?
}

main() {
    cd "$SCRIPT_DIR" || exit 1

    log_info "启动 GPU 显存监控"
    log_info "监控 GPU: $TARGET_GPUS"
    log_info "触发条件: GPU ${TARGET_GPUS} 空闲显存都 > ${MEMORY_THRESHOLD_GB}GB"
    log_info "检查间隔: ${CHECK_INTERVAL}s"
    log_info "训练脚本: $TRAIN_SCRIPT"

    check_dependencies

    while true; do
        if ! collect_target_gpu_free_memory; then
            log_warning "无法获取 GPU ${TARGET_GPUS} 的空闲显存，${CHECK_INTERVAL}s 后重试"
            sleep "$CHECK_INTERVAL"
            continue
        fi

        local all_above_threshold=1
        local status_parts=()
        local gpu_id
        local free_memory_mb
        local free_memory_gb

        for gpu_id in "${target_gpu_array[@]}"; do
            free_memory_mb="${gpu_free_memory_map[$gpu_id]}"
            free_memory_gb=$(format_mb_to_gb "$free_memory_mb")
            status_parts+=("GPU ${gpu_id}: ${free_memory_gb}GB")

            if [ "$free_memory_mb" -le "$threshold_mb" ]; then
                all_above_threshold=0
            fi
        done

        local status_summary
        status_summary=$(printf "%s, " "${status_parts[@]}")
        status_summary=${status_summary%, }

        if [ "$all_above_threshold" -eq 1 ]; then
            log_success "${status_summary}，满足条件，开始训练"
            run_training
            exit $?
        fi

        log_info "${status_summary}，继续等待"
        sleep "$CHECK_INTERVAL"
    done
}

main "$@"
