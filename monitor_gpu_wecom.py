#!/usr/bin/env python3
"""
Monitor GPU free memory and send WeCom webhook notification when condition holds.

Default behavior:
- Watch GPUs: 3,4,5
- Threshold: free memory >= 50 GB
- Sustain time: 120 seconds
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests

# WECOM_WEBHOOK = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=34a8d777-04a4-41fb-a0e8-30189d296414"

STOP = False


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(level: str, msg: str) -> None:
    print(f"[{now_str()}] {level}: {msg}", flush=True)


def parse_gpu_list(raw: str) -> List[int]:
    try:
        gpu_ids = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid GPU list: {raw}") from exc
    if not gpu_ids:
        raise argparse.ArgumentTypeError("GPU list cannot be empty")
    if any(g < 0 for g in gpu_ids):
        raise argparse.ArgumentTypeError("GPU id must be >= 0")
    return gpu_ids


def query_gpu_free_memory_mb() -> Dict[int, int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")

    free_map: Dict[int, int] = {}
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        gpu_id = int(parts[0])
        free_mb = int(parts[1])
        free_map[gpu_id] = free_mb
    return free_map


def send_wecom(webhook: str, msg: str, dry_run: bool = False) -> None:
    if dry_run:
        log("DRYRUN", f"WeCom message:\n{msg}")
        return

    payload = {"msgtype": "text", "text": {"content": msg}}
    try:
        resp = requests.post(webhook, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        errcode = data.get("errcode", -1)
        if errcode != 0:
            raise RuntimeError(f"errcode={errcode}, errmsg={data.get('errmsg')}")
        log("INFO", "WeCom notification sent successfully")
    except Exception as exc:
        log("WARN", f"WeCom send failed: {exc}")


def handle_signal(signum: int, _frame: object) -> None:
    global STOP
    STOP = True
    log("INFO", f"Received signal {signum}, stopping...")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Monitor GPUs and notify via WeCom webhook when free memory is high."
    )
    parser.add_argument("--gpus", type=parse_gpu_list, default=[2, 3, 4], help="GPU list, e.g. 3,4,5")
    parser.add_argument("--threshold-gb", type=float, default=45.0, help="Free memory threshold in GB")
    parser.add_argument("--sustain-seconds", type=int, default=120, help="Condition must hold for this many seconds")
    parser.add_argument("--interval-seconds", type=int, default=10, help="Polling interval in seconds")
    parser.add_argument("--hostname", default=os.uname().nodename, help="Name shown in notification")
    parser.add_argument("--webhook", default=os.getenv("WECOM_WEBHOOK", "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=34a8d777-04a4-41fb-a0e8-30189d296414"), help="WeCom webhook URL")
    parser.add_argument("--dry-run", action="store_true", help="Do not send real webhook, only print messages")
    args = parser.parse_args()

    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be > 0")
    if args.sustain_seconds <= 0:
        parser.error("--sustain-seconds must be > 0")
    if args.threshold_gb <= 0:
        parser.error("--threshold-gb must be > 0")
    if not args.dry_run and not args.webhook:
        parser.error("Provide --webhook or set WECOM_WEBHOOK (unless --dry-run)")

    if not shutil_which("nvidia-smi"):
        log("ERROR", "nvidia-smi not found. Please ensure NVIDIA driver is installed.")
        return 1

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    watched_gpus = args.gpus
    above_since: Dict[int, Optional[float]] = {g: None for g in watched_gpus}
    notified: Dict[int, bool] = {g: False for g in watched_gpus}

    log(
        "INFO",
        (
            f"Start monitoring GPUs={watched_gpus}, threshold={args.threshold_gb}GB, "
            f"sustain={args.sustain_seconds}s, interval={args.interval_seconds}s"
        ),
    )

    while not STOP:
        try:
            free_map = query_gpu_free_memory_mb()
        except Exception as exc:
            log("WARN", f"Failed to query GPU memory: {exc}")
            time.sleep(args.interval_seconds)
            continue

        missing = [g for g in watched_gpus if g not in free_map]
        if missing:
            log("WARN", f"GPU id not found on this machine: {missing}")

        now_ts = time.time()
        for gpu_id in watched_gpus:
            if gpu_id not in free_map:
                above_since[gpu_id] = None
                notified[gpu_id] = False
                continue

            free_gb = free_map[gpu_id] / 1024.0
            is_above = free_gb >= args.threshold_gb
            status = ">=" if is_above else "<"
            log("INFO", f"GPU {gpu_id}: free={free_gb:.2f}GB ({status} {args.threshold_gb}GB)")

            if is_above:
                if above_since[gpu_id] is None:
                    above_since[gpu_id] = now_ts
                    notified[gpu_id] = False
                elapsed = now_ts - (above_since[gpu_id] or now_ts)
                if (not notified[gpu_id]) and elapsed >= args.sustain_seconds:
                    msg = (
                        "【GPU空闲提醒】\n"
                        f"host={args.hostname}\n"
                        f"gpu={gpu_id}\n"
                        f"free_memory={free_gb:.2f}GB\n"
                        f"threshold={args.threshold_gb}GB\n"
                        f"duration>={args.sustain_seconds}s\n"
                        f"time={now_str()}"
                    )
                    send_wecom(args.webhook, msg, dry_run=args.dry_run)
                    notified[gpu_id] = True
            else:
                above_since[gpu_id] = None
                notified[gpu_id] = False

        time.sleep(args.interval_seconds)

    log("INFO", "Monitor stopped")
    return 0


def shutil_which(cmd: str) -> Optional[str]:
    for path in os.getenv("PATH", "").split(os.pathsep):
        full = os.path.join(path, cmd)
        if os.path.isfile(full) and os.access(full, os.X_OK):
            return full
    return None


if __name__ == "__main__":
    sys.exit(main())
