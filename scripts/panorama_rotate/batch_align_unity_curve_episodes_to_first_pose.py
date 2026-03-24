import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
    if (parent / "conver_equi_cube.py").is_file():
        repo_root = parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        break
else:
    raise ModuleNotFoundError(
        f"Could not locate repo root containing conver_equi_cube.py from {SCRIPT_DIR}"
    )

from batch_align_equirect_to_first_pose import (
    load_camera_poses,
    save_updated_pose_file,
    to_uint8_image,
)
"""
批量将 unity_curve_512 数据集中每个 episode 的全景图帧旋转对齐到第一帧的相机姿态，并保存旋转后的全景图和更新后的相机姿态文件到每个 episode 目录下的一个新的结果子目录中。
旋转使用 equirectangular 图像旋转函数 rotate_equirect，旋转角度由每帧相机姿态与第一帧相机姿态的欧拉角差计算得到。
支持跳过已经存在完整输出的 episode。处理过程支持多线程或多进程加速
"""


def get_rotate_equirect():
    from conver_equi_cube import rotate_equirect

    return rotate_equirect


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-align all panorama frames in unity_curve_512 episodes to the first "
            "frame pose and save results into a sibling result folder inside each "
            "episode directory."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data2/songcx/dataset/evoworld/unity_curve_512"),
        help="Dataset root containing split folders such as train/ val/ test/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Split folders to process under --dataset-root.",
    )
    parser.add_argument(
        "--episode-glob",
        type=str,
        default="episode_*",
        help="Glob pattern used to discover episode directories inside each split.",
    )
    parser.add_argument(
        "--result-dir-name",
        type=str,
        default="aligned_to_first",
        help=(
            "Name of the per-episode result directory. Output will be saved to "
            "<episode-dir>/<result-dir-name>/panorama and "
            "<episode-dir>/<result-dir-name>/camera_poses.txt."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear"],
        help="Interpolation mode passed to rotate_equirect.",
    )
    parser.add_argument(
        "--stack-original",
        action="store_true",
        help="Save a vertical stack with original on top and rotated output on bottom.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip episodes only when the output pose file exists and the output "
            "panorama frame set exactly matches the input panorama frame set."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many episodes to process in total.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when an episode fails instead of continuing.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help=(
            "Number of workers for parallel episode processing. Use 1 to disable "
            "parallelism."
        ),
    )
    parser.add_argument(
        "--executor",
        type=str,
        default="process",
        choices=["process", "thread"],
        help=(
            "Parallel executor type. 'process' is recommended for CPU-heavy image "
            "rotation; 'thread' is available as a fallback."
        ),
    )
    return parser.parse_args()


def iter_episode_dirs(dataset_root: Path, splits: Sequence[str], episode_glob: str) -> Iterable[Path]:
    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            print(f"[WARN] Split directory not found, skipping: {split_dir}")
            continue

        for episode_dir in sorted(split_dir.glob(episode_glob)):
            if episode_dir.is_dir():
                yield episode_dir


def get_frame_id(frame_path: Path) -> int:
    try:
        return int(frame_path.stem)
    except ValueError as exc:
        raise ValueError(f"Frame filename is not numeric: {frame_path}") from exc


def get_sorted_frame_paths(frame_dir: Path) -> list[Path]:
    frame_paths = [path for path in frame_dir.glob("*.png") if path.is_file()]
    frame_paths.sort(key=get_frame_id)
    return frame_paths


def has_complete_existing_output(
    input_frame_paths: Sequence[Path],
    output_dir: Path,
    output_pose_file: Path,
) -> bool:
    if not output_dir.is_dir() or not output_pose_file.is_file():
        return False

    output_frame_paths = get_sorted_frame_paths(output_dir)
    if len(output_frame_paths) != len(input_frame_paths):
        return False

    input_names = [path.name for path in input_frame_paths]
    output_names = [path.name for path in output_frame_paths]
    return input_names == output_names


def process_episode(
    episode_dir: Path,
    result_dir_name: str,
    mode: str,
    stack_original: bool,
    skip_existing: bool,
):
    rotate_equirect = get_rotate_equirect()
    panorama_dir = episode_dir / "panorama"
    pose_file = episode_dir / "camera_poses.txt"
    result_dir = episode_dir / result_dir_name
    output_dir = result_dir / "panorama"
    output_pose_file = result_dir / "camera_poses.txt"

    if not panorama_dir.is_dir():
        raise FileNotFoundError(f"Panorama directory not found: {panorama_dir}")
    if not pose_file.is_file():
        raise FileNotFoundError(f"Pose file not found: {pose_file}")

    poses = load_camera_poses(pose_file)
    frame_paths = get_sorted_frame_paths(panorama_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No PNG files found in: {panorama_dir}")

    if skip_existing and has_complete_existing_output(
        input_frame_paths=frame_paths,
        output_dir=output_dir,
        output_pose_file=output_pose_file,
    ):
        return 0, True

    first_frame_id = get_frame_id(frame_paths[0])
    if first_frame_id not in poses:
        raise KeyError(
            f"Missing pose for first image frame {first_frame_id}: {frame_paths[0]}"
        )
    reference_pose = poses[first_frame_id]

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for frame_path in frame_paths:
        frame_id = get_frame_id(frame_path)
        if frame_id not in poses:
            raise KeyError(f"Missing pose for frame {frame_id}: {frame_path}")

        pose = poses[frame_id]
        yaw_deg = pose["roty"] - reference_pose["roty"]
        pitch_deg = pose["rotx"] - reference_pose["rotx"]
        roll_deg = pose["rotz"] - reference_pose["rotz"]

        image = np.array(Image.open(frame_path).convert("RGB"))
        rotated = rotate_equirect(
            image,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            mode=mode,
        )

        original_uint8 = to_uint8_image(image)
        rotated_uint8 = to_uint8_image(rotated)
        output_uint8 = (
            np.concatenate([original_uint8, rotated_uint8], axis=0)
            if stack_original
            else rotated_uint8
        )

        Image.fromarray(output_uint8).save(output_dir / frame_path.name)
        processed += 1

    save_updated_pose_file(output_pose_file, poses, reference_pose)
    return processed, False


def process_episode_task(task: tuple[Path, str, str, bool, bool]):
    episode_dir, result_dir_name, mode, stack_original, skip_existing = task
    processed_frames, skipped = process_episode(
        episode_dir=episode_dir,
        result_dir_name=result_dir_name,
        mode=mode,
        stack_original=stack_original,
        skip_existing=skip_existing,
    )
    return episode_dir, processed_frames, skipped


def main():
    args = parse_args()

    episode_dirs = list(iter_episode_dirs(args.dataset_root, args.splits, args.episode_glob))
    if args.limit is not None:
        episode_dirs = episode_dirs[: args.limit]

    attempted = len(episode_dirs)
    processed_episodes = 0
    skipped_episodes = 0
    failed_episodes = []
    total_frames = 0

    tasks = [
        (
            episode_dir,
            args.result_dir_name,
            args.mode,
            args.stack_original,
            args.skip_existing,
        )
        for episode_dir in episode_dirs
    ]

    if args.num_workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            episode_dir = task[0]
            try:
                _, processed_frames, skipped = process_episode_task(task)
                if skipped:
                    skipped_episodes += 1
                    print(f"[SKIP] {episode_dir}")
                    continue

                processed_episodes += 1
                total_frames += processed_frames
                print(
                    f"[OK] {episode_dir} -> "
                    f"{episode_dir / args.result_dir_name} ({processed_frames} frames)"
                )
            except Exception as exc:
                failed_episodes.append((episode_dir, str(exc)))
                print(f"[FAIL] {episode_dir}: {exc}")
                if args.fail_fast:
                    raise
    else:
        executor_cls = (
            ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
        )
        print(
            f"Running with {args.executor} executor, "
            f"{args.num_workers} workers, {len(tasks)} episodes."
        )

        with executor_cls(max_workers=args.num_workers) as executor:
            future_to_episode = {
                executor.submit(process_episode_task, task): task[0] for task in tasks
            }

            try:
                for future in as_completed(future_to_episode):
                    episode_dir = future_to_episode[future]
                    try:
                        _, processed_frames, skipped = future.result()
                        if skipped:
                            skipped_episodes += 1
                            print(f"[SKIP] {episode_dir}")
                            continue

                        processed_episodes += 1
                        total_frames += processed_frames
                        print(
                            f"[OK] {episode_dir} -> "
                            f"{episode_dir / args.result_dir_name} ({processed_frames} frames)"
                        )
                    except Exception as exc:
                        failed_episodes.append((episode_dir, str(exc)))
                        print(f"[FAIL] {episode_dir}: {exc}")
                        if args.fail_fast:
                            for pending_future in future_to_episode:
                                pending_future.cancel()
                            raise
            except Exception:
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    print()
    print("Batch alignment finished.")
    print(f"Attempted episodes: {attempted}")
    print(f"Processed episodes: {processed_episodes}")
    print(f"Skipped episodes: {skipped_episodes}")
    print(f"Failed episodes: {len(failed_episodes)}")
    print(f"Total processed frames: {total_frames}")

    if failed_episodes:
        print("Failed episode list:")
        for episode_dir, error_msg in failed_episodes:
            print(f"  - {episode_dir}: {error_msg}")


if __name__ == "__main__":
    main()
