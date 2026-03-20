#!/bin/bash

# GPU显存监控和自动训练脚本
# 监控GPU 2号卡的空闲显存，当空闲显存连续3分钟都大于50GB时执行 train_pretrain.sh

TARGET_GPU=2
MEMORY_THRESHOLD_GB=58
SUSTAIN_SECONDS=180
CHECK_INTERVAL=10
TRAIN_SCRIPT="train_pretrain.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

threshold_mb=$((MEMORY_THRESHOLD_GB * 1024))
above_threshold_since=""

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

check_dependencies() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_error "nvidia-smi 未找到，请确保已安装 NVIDIA 驱动"
        exit 1
    fi

    if [ ! -f "$TRAIN_SCRIPT" ]; then
        log_error "训练脚本 '$TRAIN_SCRIPT' 未找到"
        exit 1
    fi

    local gpu_count
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    if [ "$TARGET_GPU" -ge "$gpu_count" ]; then
        log_error "GPU $TARGET_GPU 不存在，当前机器共有 $gpu_count 张 GPU"
        exit 1
    fi
}

get_gpu_free_memory_mb() {
    nvidia-smi \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits \
        --id="$TARGET_GPU" 2>/dev/null
}

format_mb_to_gb() {
    awk "BEGIN { printf \"%.2f\", $1 / 1024 }"
}

run_training() {
    log_success "GPU $TARGET_GPU 空闲显存已连续 ${SUSTAIN_SECONDS}s 大于 ${MEMORY_THRESHOLD_GB}GB，开始执行 $TRAIN_SCRIPT"
    bash "$TRAIN_SCRIPT"
    local exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        log_success "训练脚本执行完成"
    else
        log_error "训练脚本退出异常，退出码: $exit_code"
    fi

    return "$exit_code"
}

cleanup() {
    log_info "收到退出信号，停止监控"
    exit 0
}

trap cleanup SIGINT SIGTERM

main() {
    log_info "启动 GPU 显存监控"
    log_info "监控 GPU: $TARGET_GPU"
    log_info "触发条件: 空闲显存持续 ${SUSTAIN_SECONDS}s 大于 ${MEMORY_THRESHOLD_GB}GB"
    log_info "检查间隔: ${CHECK_INTERVAL}s"
    log_info "训练脚本: $TRAIN_SCRIPT"

    check_dependencies

    while true; do
        local free_memory_mb
        free_memory_mb=$(get_gpu_free_memory_mb)

        if [ -z "$free_memory_mb" ] || ! [[ "$free_memory_mb" =~ ^[0-9]+$ ]]; then
            log_warning "无法获取 GPU $TARGET_GPU 的空闲显存，${CHECK_INTERVAL}s 后重试"
            sleep "$CHECK_INTERVAL"
            continue
        fi

        local free_memory_gb
        free_memory_gb=$(format_mb_to_gb "$free_memory_mb")

        if [ "$free_memory_mb" -gt "$threshold_mb" ]; then
            local now_ts
            now_ts=$(date +%s)

            if [ -z "$above_threshold_since" ]; then
                above_threshold_since="$now_ts"
                log_success "GPU $TARGET_GPU 空闲显存 ${free_memory_gb}GB > ${MEMORY_THRESHOLD_GB}GB，开始计时"
            else
                local elapsed
                elapsed=$((now_ts - above_threshold_since))
                log_info "GPU $TARGET_GPU 空闲显存 ${free_memory_gb}GB > ${MEMORY_THRESHOLD_GB}GB，已持续 ${elapsed}s/${SUSTAIN_SECONDS}s"

                if [ "$elapsed" -ge "$SUSTAIN_SECONDS" ]; then
                    run_training
                    exit $?
                fi
            fi
        else
            if [ -n "$above_threshold_since" ]; then
                log_warning "GPU $TARGET_GPU 空闲显存回落到 ${free_memory_gb}GB，低于或等于 ${MEMORY_THRESHOLD_GB}GB，重置计时"
                above_threshold_since=""
            else
                log_info "GPU $TARGET_GPU 空闲显存 ${free_memory_gb}GB，继续等待"
            fi
        fi

        sleep "$CHECK_INTERVAL"
    done
}

main "$@"
