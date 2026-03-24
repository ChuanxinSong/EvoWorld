from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
    if (parent / "images_to_mp4.py").is_file():
        repo_root = parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        break
else:
    raise ModuleNotFoundError(
        f"Could not locate repo root containing images_to_mp4.py from {SCRIPT_DIR}"
    )

from images_to_mp4 import save_video as save_video_with_imageio

"""
可视化每个 episode 中原始全景图和旋转后全景图的对齐情况，生成竖直拼接的对比视频，方便人工检查旋转结果是否正确。
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "For each unity_curve_512 episode, vertically stack the original panorama "
            "and the rotated panorama, draw labels, and save an MP4 video."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("unity_curve_512"),
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
            "Per-episode result directory containing rotated panoramas. "
            "The video and stacked frames are also written into this directory."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional output root for stacked frames and MP4 files. When omitted, "
            "outputs are written back into <episode>/<result-dir-name>/."
        ),
    )
    parser.add_argument(
        "--original-dir-name",
        type=str,
        default="panorama",
        help="Subdirectory under each episode containing original panoramas.",
    )
    parser.add_argument(
        "--rotated-dir-name",
        type=str,
        default="panorama",
        help=(
            "Subdirectory under <episode>/<result-dir-name>/ containing rotated "
            "panoramas."
        ),
    )
    parser.add_argument(
        "--stacked-dir-name",
        type=str,
        default="panorama_original_vs_rotated",
        help="Directory name for labeled stacked PNG frames inside result dir.",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default="panorama_original_vs_rotated.mp4",
        help="Output MP4 filename inside result dir.",
    )
    parser.add_argument(
        "--top-label",
        type=str,
        default="original",
        help="Label drawn on the top panorama.",
    )
    parser.add_argument(
        "--bottom-label",
        type=str,
        default="rotated",
        help="Label drawn on the bottom panorama.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second for the output video.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 CRF quality parameter. Lower means higher quality.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip an episode when the stacked PNG frames exactly match the input "
            "frame list and the output MP4 already exists."
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
    output_video_path: Path,
) -> bool:
    if not output_dir.is_dir() or not output_video_path.is_file():
        return False

    output_frame_paths = get_sorted_frame_paths(output_dir)
    if len(output_frame_paths) != len(input_frame_paths):
        return False

    input_names = [path.name for path in input_frame_paths]
    output_names = [path.name for path in output_frame_paths]
    return input_names == output_names


def load_font(font_size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def draw_label(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    padding: int,
    outline_width: int,
):
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle(
        (left - padding, top - padding, right + padding, bottom + padding),
        fill=(0, 0, 0),
        outline=(255, 255, 255),
        width=outline_width,
    )
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def pad_image_to_even_size(image: Image.Image) -> Image.Image:
    width, height = image.size
    pad_w = width % 2
    pad_h = height % 2
    if pad_w == 0 and pad_h == 0:
        return image

    padded = Image.new("RGB", (width + pad_w, height + pad_h), color=(0, 0, 0))
    padded.paste(image, (0, 0))
    return padded


def build_labeled_stack(
    original_path: Path,
    rotated_path: Path,
    top_label: str,
    bottom_label: str,
) -> Image.Image:
    original = Image.open(original_path).convert("RGB")
    rotated = Image.open(rotated_path).convert("RGB")

    if original.size != rotated.size:
        raise ValueError(
            "Original and rotated panorama sizes differ: "
            f"{original_path}={original.size}, {rotated_path}={rotated.size}"
        )

    width, height = original.size
    stacked = Image.new("RGB", (width, height * 2))
    stacked.paste(original, (0, 0))
    stacked.paste(rotated, (0, height))

    font_size = max(18, height // 16)
    margin = max(12, font_size // 2)
    padding = max(6, font_size // 4)
    outline_width = max(1, font_size // 12)
    font = load_font(font_size)

    draw = ImageDraw.Draw(stacked)
    draw_label(draw, margin, margin, top_label, font, padding, outline_width)
    draw_label(
        draw,
        margin,
        height + margin,
        bottom_label,
        font,
        padding,
        outline_width,
    )

    return pad_image_to_even_size(stacked)


def pil_image_to_uint8_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_video_with_cv2(
    video_frames: Sequence[Image.Image],
    output_video_path: Path,
    fps: int,
):
    if not video_frames:
        raise ValueError("video_frames is empty")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame = pil_image_to_uint8_array(video_frames[0])
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_video_path}")

    try:
        for frame in video_frames:
            frame_uint8 = pil_image_to_uint8_array(frame)
            if frame_uint8.shape[:2] != (height, width):
                raise ValueError(
                    "All video frames must have the same size. "
                    f"Expected {(width, height)}, got {frame_uint8.shape[1::-1]}"
                )
            writer.write(cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def save_video_robust(
    video_frames: Sequence[Image.Image],
    output_video_path: Path,
    fps: int,
    crf: int,
):
    try:
        save_video_with_imageio(
            video_frames=video_frames,
            output_video_path=str(output_video_path),
            fps=fps,
            crf=crf,
        )
    except Exception as exc:
        print(f"[WARN] imageio video writer failed for {output_video_path}: {exc}")
        save_video_with_cv2(video_frames=video_frames, output_video_path=output_video_path, fps=fps)


def get_output_result_dir(args, episode_dir: Path) -> Path:
    if args.output_root is None:
        return episode_dir / args.result_dir_name

    episode_rel = episode_dir.relative_to(args.dataset_root)
    return args.output_root / episode_rel / args.result_dir_name


def process_episode(args, episode_dir: Path) -> tuple[int, bool, Path]:
    original_dir = episode_dir / args.original_dir_name
    rotated_result_dir = episode_dir / args.result_dir_name
    rotated_dir = rotated_result_dir / args.rotated_dir_name
    output_result_dir = get_output_result_dir(args, episode_dir)
    stacked_dir = output_result_dir / args.stacked_dir_name
    video_path = output_result_dir / args.video_name

    if not original_dir.is_dir():
        raise FileNotFoundError(f"Original panorama directory not found: {original_dir}")
    if not rotated_dir.is_dir():
        raise FileNotFoundError(f"Rotated panorama directory not found: {rotated_dir}")

    original_frame_paths = get_sorted_frame_paths(original_dir)
    rotated_frame_paths = get_sorted_frame_paths(rotated_dir)
    if not original_frame_paths:
        raise FileNotFoundError(f"No PNG files found in: {original_dir}")
    if not rotated_frame_paths:
        raise FileNotFoundError(f"No PNG files found in: {rotated_dir}")

    original_names = [path.name for path in original_frame_paths]
    rotated_names = [path.name for path in rotated_frame_paths]
    if original_names != rotated_names:
        raise ValueError(
            "Original and rotated frame lists do not match exactly.\n"
            f"Original count: {len(original_names)} in {original_dir}\n"
            f"Rotated count: {len(rotated_names)} in {rotated_dir}"
        )

    if args.skip_existing and has_complete_existing_output(
        input_frame_paths=original_frame_paths,
        output_dir=stacked_dir,
        output_video_path=video_path,
    ):
        return len(original_frame_paths), True, video_path

    stacked_dir.mkdir(parents=True, exist_ok=True)

    video_frames = []
    for original_path, rotated_path in zip(original_frame_paths, rotated_frame_paths):
        stacked_frame = build_labeled_stack(
            original_path=original_path,
            rotated_path=rotated_path,
            top_label=args.top_label,
            bottom_label=args.bottom_label,
        )
        stacked_frame.save(stacked_dir / original_path.name)
        video_frames.append(stacked_frame)

    save_video_robust(
        video_frames=video_frames,
        output_video_path=video_path,
        fps=args.fps,
        crf=args.crf,
    )
    return len(video_frames), False, video_path


def main():
    args = parse_args()

    episode_dirs = list(iter_episode_dirs(args.dataset_root, args.splits, args.episode_glob))
    if args.limit is not None:
        episode_dirs = episode_dirs[: args.limit]

    attempted = len(episode_dirs)
    processed_episodes = 0
    skipped_episodes = 0
    failed_episodes: list[tuple[Path, str]] = []
    total_frames = 0

    for episode_dir in episode_dirs:
        try:
            processed_frames, skipped, video_path = process_episode(args, episode_dir)
            if skipped:
                skipped_episodes += 1
                print(f"[SKIP] {episode_dir} -> {video_path}")
                continue

            processed_episodes += 1
            total_frames += processed_frames
            print(f"[OK] {episode_dir} -> {video_path} ({processed_frames} frames)")
        except Exception as exc:
            failed_episodes.append((episode_dir, str(exc)))
            print(f"[FAIL] {episode_dir}: {exc}")
            if args.fail_fast:
                raise

    print()
    print("Batch video generation finished.")
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
