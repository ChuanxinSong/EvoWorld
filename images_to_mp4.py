import argparse
from pathlib import Path
from typing import List, Union

import imageio.v2 as imageio
import numpy as np
import PIL.Image


def save_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]],
    output_video_path: str,
    fps: int = 10,
    crf: int = 18,
) -> str:
    if not video_frames:
        raise ValueError("video_frames is empty")

    if isinstance(video_frames[0], np.ndarray):
        converted_frames = []
        for frame in video_frames:
            if np.issubdtype(frame.dtype, np.floating):
                max_value = float(np.nanmax(frame)) if frame.size else 0.0
                if max_value <= 1.0:
                    frame = frame * 255.0
            converted_frames.append(np.clip(frame, 0, 255).astype(np.uint8))
        video_frames = converted_frames
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    else:
        raise TypeError("video_frames must contain numpy arrays or PIL images")

    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-crf", str(crf)],
    )
    try:
        for frame in video_frames:
            writer.append_data(frame)
    finally:
        writer.close()

    return str(output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an ordered image sequence into an MP4 video."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("equi_rotate_results/test/episode_0001/panorama_aligned_to_first"),
        help="Directory containing ordered PNG frames.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("equi_rotate_results/test/episode_0001/panorama_aligned_to_first.mp4"),
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 CRF quality parameter. Lower means higher quality.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    image_paths = sorted(args.input_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in: {args.input_dir}")

    video_frames = [PIL.Image.open(path).convert("RGB") for path in image_paths]
    output_path = save_video(
        video_frames=video_frames,
        output_video_path=str(args.output),
        fps=args.fps,
        crf=args.crf,
    )
    print(f"Saved video to: {output_path}")
    print(f"Frame count: {len(video_frames)}")


if __name__ == "__main__":
    main()
