import argparse
import csv
import sys
from pathlib import Path

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

"""
给定一个 episode 目录，批量将其中的全景图帧旋转对齐到第一帧的相机姿态，并保存旋转后的全景图和更新后的相机姿态文件到一个新的结果子目录中。
旋转使用 equirectangular 图像旋转函数 rotate_equirect，旋转角度由每帧相机姿态与第一帧相机姿态的欧拉角差计算得到。
"""

POSE_HEADER = ["Frame", "PosX", "PosY", "PosZ", "RotX", "RotY", "RotZ"]


def get_rotate_equirect():
    from conver_equi_cube import rotate_equirect

    return rotate_equirect


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rotate all panorama frames so their orientation matches the first frame pose, "
            "while keeping per-frame positions unchanged."
        )
    )
    parser.add_argument(
        "--episode-dir",
        type=Path,
        default=Path("unity_curve/test/episode_0001"),
        help="Episode directory containing panorama/ and camera_poses.txt.",
    )
    parser.add_argument(
        "--panorama-dir",
        type=Path,
        default=None,
        help="Optional panorama directory override. Defaults to <episode-dir>/panorama.",
    )
    parser.add_argument(
        "--pose-file",
        type=Path,
        default=None,
        help="Optional pose file override. Defaults to <episode-dir>/camera_poses.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <episode-dir>/panorama_aligned_to_first.",
    )
    parser.add_argument(
        "--output-pose-file",
        type=Path,
        default=None,
        help="Optional output pose file. Defaults to <episode-dir>/camera_poses_aligned_to_first.txt.",
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
    return parser.parse_args()


def normalize_key(key: str) -> str:
    return key.strip().lower().replace(".", "").replace("_", "")


def get_default_output_base(episode_dir: Path) -> Path:
    parent_name = episode_dir.parent.name or "episode_group"
    episode_name = episode_dir.name or "episode"
    return Path("equi_rotate_results") / parent_name / episode_name


def load_camera_poses(pose_file: Path):
    with pose_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Pose file has no header: {pose_file}")

        name_map = {normalize_key(name): name for name in reader.fieldnames}

        required_aliases = {
            "frame": ["frame"],
            "posx": ["posx", "positionx"],
            "posy": ["posy", "positiony"],
            "posz": ["posz", "positionz"],
            "rotx": ["rotx", "rotationx"],
            "roty": ["roty", "rotationy"],
            "rotz": ["rotz", "rotationz"],
        }

        resolved = {}
        for target, aliases in required_aliases.items():
            for alias in aliases:
                if alias in name_map:
                    resolved[target] = name_map[alias]
                    break
            else:
                raise KeyError(
                    f"Missing required column for {target} in pose file: {pose_file}"
                )

        poses = {}
        for row in reader:
            frame_id = int(float(row[resolved["frame"]]))
            poses[frame_id] = {
                "frame": frame_id,
                "posx": float(row[resolved["posx"]]),
                "posy": float(row[resolved["posy"]]),
                "posz": float(row[resolved["posz"]]),
                "rotx": float(row[resolved["rotx"]]),
                "roty": float(row[resolved["roty"]]),
                "rotz": float(row[resolved["rotz"]]),
            }

    if not poses:
        raise ValueError(f"No poses found in: {pose_file}")

    return poses


def to_uint8_image(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img

    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        max_value = float(np.nanmax(img)) if img.size else 0.0
        if max_value <= 1.0:
            img = img * 255.0

    return np.clip(img, 0, 255).astype(np.uint8)


def save_updated_pose_file(output_pose_file: Path, poses: dict, reference_pose: dict):
    with output_pose_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(POSE_HEADER)
        for frame_id in sorted(poses):
            pose = poses[frame_id]
            writer.writerow(
                [
                    frame_id,
                    pose["posx"],
                    pose["posy"],
                    pose["posz"],
                    reference_pose["rotx"],
                    reference_pose["roty"],
                    reference_pose["rotz"],
                ]
            )


def main():
    args = parse_args()
    rotate_equirect = get_rotate_equirect()

    episode_dir = args.episode_dir
    panorama_dir = args.panorama_dir or (episode_dir / "panorama")
    pose_file = args.pose_file or (episode_dir / "camera_poses.txt")
    default_output_base = get_default_output_base(episode_dir)
    output_dir = args.output_dir or (default_output_base / "panorama_aligned_to_first")
    output_pose_file = args.output_pose_file or (
        default_output_base / "camera_poses_aligned_to_first.txt"
    )

    if not panorama_dir.exists():
        raise FileNotFoundError(f"Panorama directory not found: {panorama_dir}")
    if not pose_file.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_file}")

    poses = load_camera_poses(pose_file)
    frame_paths = sorted(panorama_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No PNG files found in: {panorama_dir}")

    first_frame_id = min(poses)
    if first_frame_id not in poses:
        raise KeyError(f"First frame pose not found: {first_frame_id}")
    reference_pose = poses[first_frame_id]

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for frame_path in frame_paths:
        frame_id = int(frame_path.stem)
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
            mode=args.mode,
        )

        original_uint8 = to_uint8_image(image)
        rotated_uint8 = to_uint8_image(rotated)
        output_uint8 = (
            np.concatenate([original_uint8, rotated_uint8], axis=0)
            if args.stack_original
            else rotated_uint8
        )

        Image.fromarray(output_uint8).save(output_dir / frame_path.name)
        processed += 1

    save_updated_pose_file(output_pose_file, poses, reference_pose)

    print(f"Processed {processed} frames")
    print(f"Saved aligned panoramas to: {output_dir}")
    print(f"Saved aligned poses to: {output_pose_file}")
    print(
        "Reference pose: "
        f"RotX={reference_pose['rotx']}, "
        f"RotY={reference_pose['roty']}, "
        f"RotZ={reference_pose['rotz']}"
    )


if __name__ == "__main__":
    main()
