#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from evoworld.pipeline.pipeline_evoworld import StableVideoDiffusionPipeline
from evoworld.trainer.unet_plucker import UNetSpatioTemporalConditionModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assemble a full Stable Video Diffusion pipeline from a trained UNet checkpoint."
    )
    parser.add_argument(
        "--unet_path",
        required=True,
        help="Path to a checkpoint root containing `unet/`, or the `unet/` directory itself.",
    )
    parser.add_argument(
        "--svd_path",
        default="stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        help="Base Stable Video Diffusion model path or Hugging Face model id.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save the assembled pipeline. Defaults to `<unet_path>_pipeline`.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only load model files from local cache/disk.",
    )
    return parser.parse_args()


def resolve_unet_root_and_subfolder(unet_path: str):
    path = Path(unet_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"UNet path does not exist: {path}")

    if path.is_dir() and (path / "config.json").exists():
        return str(path.parent), path.name

    if path.is_dir() and (path / "unet" / "config.json").exists():
        return str(path), "unet"

    raise FileNotFoundError(
        "Could not find a valid UNet config. Expected either `<path>/config.json` "
        f"or `<path>/unet/config.json`, got: {path}"
    )


def default_output_dir(unet_path: str) -> str:
    path = Path(unet_path).expanduser().resolve()
    if path.name == "unet":
        return str(path.parent.parent / f"{path.parent.name}_pipeline")
    return str(path.parent / f"{path.name}_pipeline")


def main():
    args = parse_args()

    unet_root, unet_subfolder = resolve_unet_root_and_subfolder(args.unet_path)
    output_dir = args.output_dir or default_output_dir(args.unet_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading UNet from: root={unet_root}, subfolder={unet_subfolder}")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_root,
        subfolder=unet_subfolder,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )

    print(f"Loading base pipeline from: {args.svd_path}")
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.svd_path,
        unet=unet,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )

    print(f"Saving assembled pipeline to: {output_dir}")
    pipeline.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
