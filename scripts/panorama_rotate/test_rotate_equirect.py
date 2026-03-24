import argparse
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
旋转单张全景图以验证 rotate_equirect 函数的正确性。输入图像和旋转参数可通过命令行指定，输出图像会保存到指定目录下，文件名包含旋转参数信息。
"""


def get_rotate_equirect():
    from conver_equi_cube import rotate_equirect

    return rotate_equirect


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rotate an equirectangular image with arbitrary yaw/pitch/roll."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("001.png"),
        help="Input equirectangular image path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("equi_rotate_results"),
        help="Directory to save rotated images.",
    )
    parser.add_argument(
        "--yaw",
        type=float,
        default=0.0,
        help="Yaw rotation in degrees.",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=0.0,
        help="Pitch rotation in degrees.",
    )
    parser.add_argument(
        "--roll",
        type=float,
        default=0.0,
        help="Roll rotation in degrees.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear"],
        help="Interpolation mode used by rotate_equirect.",
    )
    return parser.parse_args()


def to_uint8_image(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img

    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        max_value = float(np.nanmax(img)) if img.size else 0.0
        if max_value <= 1.0:
            img = img * 255.0

    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    args = parse_args()
    rotate_equirect = get_rotate_equirect()
    input_path = args.input
    output_dir = args.output_dir

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image = np.array(Image.open(input_path).convert("RGB"))
    rotated = rotate_equirect(
        image,
        yaw_deg=args.yaw,
        pitch_deg=args.pitch,
        roll_deg=args.roll,
        mode=args.mode,
    )
    original_uint8 = to_uint8_image(image)
    rotated_uint8 = to_uint8_image(rotated)
    stacked_uint8 = np.concatenate([original_uint8, rotated_uint8], axis=0)

    output_name = (
        f"{input_path.stem}_yaw{args.yaw:g}_pitch{args.pitch:g}_roll{args.roll:g}.png"
    )
    output_path = output_dir / output_name
    Image.fromarray(stacked_uint8).save(output_path)

    print(f"Saved rotated image to: {output_path}")


if __name__ == "__main__":
    main()
