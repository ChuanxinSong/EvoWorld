import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from conver_equi_cube import rotate_equirect
from dataset.CameraTrajDataset import (
    CameraTrajDataset,
    load_camera_poses_from_txt,
    xyz_euler_to_three_by_four_matrix_batch,
)
from evoworld.pipeline.pipeline_evoworld import StableVideoDiffusionPipeline
from evoworld.trainer.unet_plucker import UNetSpatioTemporalConditionModel
from utils.image_utils import frame_to_pil
from utils.plucker_embedding import equirectangular_to_ray, ray_c2w_to_plucker

sys.path.append("./evoworld")


def parse_arguments():
    """Parse command-line arguments for forward pass evaluation."""
    parser = argparse.ArgumentParser(description="Read safetensors file")
    parser.add_argument("--ckpt", type=str, help="Path to safetensors file")
    parser.add_argument("--step_id", type=str, help="e.g., 20000")
    parser.add_argument("--data", type=str, help="Path to data file")
    parser.add_argument("--width", type=int, help="Width of the image")
    parser.add_argument("--height", type=int, help="Height of the image")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of the image")
    parser.add_argument("--num_data", type=int, default=1, help="Num of images to test.")
    parser.add_argument(
        "--add_plucker", action="store_true", help="Whether to add plucker embeddings"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Whether to enable verbose logging"
    )
    parser.add_argument("--mask_mem", action="store_true")
    parser.add_argument("--num_frames", default=25, type=int)
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=8,
        help="VAE decode chunk size. Set to a smaller value to reduce memory usage.",
    )
    parser.add_argument(
        "--reprojection_name", type=str, default="rendered_panorama_vggt_open3d"
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="reprojection",
        choices=["reprojection", "empty_with_traj"],
        help="Memory sampling method used by CameraTrajDataset during inference.",
    )
    parser.add_argument("--output_name", type=str, default="eval_add_mem")
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def setup_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def determine_data_config(data_path: str, reprojection_name: str) -> Tuple[str, bool]:
    """Determine data root and whether it's a single video."""
    try:
        folder_contents = os.listdir(data_path)
        is_single_video = reprojection_name in folder_contents
        data_root = data_path if is_single_video else f"{data_path}/test"
        return data_root, is_single_video
    except FileNotFoundError:
        raise ValueError(f"Data path '{data_path}' not found")


def create_dataset_and_loader(args, data_root: str, is_single_video: bool, loop_args: dict):
    """Create dataset and data loader for evaluation."""
    val_dataset = CameraTrajDataset(
        data_root,
        width=args.width,
        height=args.height,
        trajectory_file=None,
        memory_sampling_args=loop_args,
        sequence_length=args.num_frames,
        last_segment_length=args.num_frames,
        reprojection_name=args.reprojection_name,
        is_single_video=is_single_video,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )

    return val_dataset, val_loader


def setup_model_and_pipeline(args):
    """Setup UNet model and pipeline for inference."""
    weight_dtype = torch.float32

    # Load UNet
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.ckpt,
        subfolder="unet",
        low_cpu_mem_usage=True,
    )
    unet.requires_grad_(False)

    # Load pipeline
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.ckpt,
        unet=unet,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=not args.verbose)

    # Setup rays for plucker embeddings
    rays = equirectangular_to_ray(target_H=args.height // 8, target_W=args.width // 8)
    rays = torch.tensor(rays).to(weight_dtype).to("cuda")

    return pipeline, rays, weight_dtype


def prepare_batch_data(batch, args, rays, weight_dtype):
    """Prepare batch data including camera trajectories and plucker embeddings."""
    images = batch["pixel_values"]
    first_frame = images[:, 0, :, :, :].to("cuda")
    camera_traj_raw = batch["cam_traj"].to("cuda")

    # Initialize camera trajectory tensor
    camera_traj = (
        torch.zeros(camera_traj_raw.shape[0], args.num_frames, 3, 4)
        .to(weight_dtype)
        .to("cuda", non_blocking=True)
    )

    # Initialize plucker embedding tensor
    plucker_embedding = (
        torch.zeros(
            camera_traj_raw.shape[0],
            args.num_frames,
            6,
            args.height // 8,
            args.width // 8,
        )
        .to(weight_dtype)
        .to("cuda", non_blocking=True)
    )

    # Compute camera trajectories and plucker embeddings
    for i in range(camera_traj.shape[0]):
        camera_traj[i] = xyz_euler_to_three_by_four_matrix_batch(
            camera_traj_raw[i], relative=True
        )  # Step, 3, 4
        plucker_embedding[i] = ray_c2w_to_plucker(
            rays, camera_traj[i]
        )  # Step, 6, 72, 128

    memorized_pixel_values = batch["memorized_pixel_values"].to("cuda")

    return first_frame, camera_traj, plucker_embedding, memorized_pixel_values, images


def save_frames(video_frames, gt_frames, frames_path: str, frames_gt_path: str, num_frames: int):
    """Save predicted, ground truth, and side-by-side comparison frames to disk."""
    os.makedirs(frames_path, exist_ok=True)
    os.makedirs(frames_gt_path, exist_ok=True)
    frames_cmp_path = frames_path.replace("predictions", "predictions_compare")
    os.makedirs(frames_cmp_path, exist_ok=True)

    assert (
        len(video_frames) == num_frames
    ), f"video frames {len(video_frames)} should equal num_frames {num_frames}!"

    for i in range(num_frames):
        frame = frame_to_pil(video_frames[i])
        gt_frame = frame_to_pil(gt_frames[i])
        if gt_frame.size != frame.size:
            gt_frame = gt_frame.resize(frame.size, Image.BICUBIC)
        comparison = Image.new("RGB", (frame.width, frame.height + gt_frame.height))
        comparison.paste(frame, (0, 0))
        comparison.paste(gt_frame, (0, frame.height))

        # Save frames
        frame.save(os.path.join(frames_path, f"{i+1:03}.png"))
        gt_frame.save(os.path.join(frames_gt_path, f"{i+1:03}.png"))
        comparison.save(os.path.join(frames_cmp_path, f"{i+1:03}.png"))


def _extract_episode_path(batch) -> str:
    episode_path = batch["episode_path"]
    if isinstance(episode_path, (list, tuple)):
        return episode_path[0]
    return episode_path


def _extract_frame_ids(batch) -> List[int]:
    frame_ids = batch["frame_ids"]
    if isinstance(frame_ids, torch.Tensor):
        if frame_ids.ndim == 2:
            frame_ids = frame_ids[0]
        return [int(x) for x in frame_ids.tolist()]
    return [int(x) for x in frame_ids]


def _load_raw_gt_frames(episode_path: str, frame_ids: List[int]) -> List[Image.Image]:
    gt_frames = []
    for frame_id in frame_ids:
        frame_path = os.path.join(episode_path, "panorama", f"{frame_id:03}.png")
        gt_frames.append(Image.open(frame_path).convert("RGB"))
    return gt_frames


def _recover_only_position_predictions(video_frames, frame_ids: List[int], episode_path: str) -> List[Image.Image]:
    raw_poses = load_camera_poses_from_txt(os.path.join(episode_path, "camera_poses.txt"))
    aligned_poses = load_camera_poses_from_txt(
        os.path.join(episode_path, "aligned_to_first", "camera_poses.txt")
    )

    recovered_frames = []
    # import pdb; pdb.set_trace()
    for frame, frame_id in zip(video_frames, frame_ids):
        frame_key = str(frame_id)
        raw_pose = raw_poses[frame_key]
        aligned_pose = aligned_poses[frame_key]

        yaw_deg = raw_pose[4] - aligned_pose[4]
        pitch_deg = raw_pose[3] - aligned_pose[3]
        roll_deg = raw_pose[5] - aligned_pose[5]

        frame_np = np.array(frame_to_pil(frame).convert("RGB"))
        recovered_np = rotate_equirect(
            frame_np,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            mode="bilinear",
        )
        recovered_frames.append(Image.fromarray(np.clip(recovered_np, 0, 255).astype(np.uint8)))

    return recovered_frames


def process_batch(
    batch,
    args,
    pipeline,
    rays,
    weight_dtype,
    output_path: str,
    episode: str,
    decode_chunk_size: int = 8,
):
    """Process a single batch for inference and save results."""
    # Prepare batch data
    first_frame, camera_traj, plucker_embedding, memorized_pixel_values, images = prepare_batch_data(
        batch, args, rays, weight_dtype
    )

    # Run inference
    with torch.inference_mode():
        video_frames = pipeline(
            first_frame,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            decode_chunk_size=decode_chunk_size,
            motion_bucket_id=127,
            fps=7,
            noise_aug_strength=0.02,
            plucker_embedding=plucker_embedding,
            memorized_pixel_values=memorized_pixel_values,
            mask_mem=args.mask_mem,
        ).frames[0]

    # Setup output paths
    frames_path = os.path.join(output_path, episode, "predictions")
    frames_gt_path = os.path.join(output_path, episode, "predictions_gt")

    if getattr(args, "only_position", False):
        episode_path = _extract_episode_path(batch)
        frame_ids = _extract_frame_ids(batch)
        video_frames = _recover_only_position_predictions(video_frames, frame_ids, episode_path)
        gt_frames = _load_raw_gt_frames(episode_path, frame_ids)
    else:
        gt_frames = images[0]

    # Save frames
    save_frames(video_frames, gt_frames, frames_path, frames_gt_path, args.num_frames)


def main():
    """Main function for forward pass evaluation."""
    args = parse_arguments()

    # Setup
    setup_random_seeds(args.seed)

    # Configuration
    loop_args = {
        "sampling_method": args.sampling_method,
        "include_initial_frame": True,
    }

    # Setup output path
    output_path = os.path.join(args.ckpt, args.output_name)
    os.makedirs(output_path, exist_ok=True)

    # Determine data configuration
    data_root, is_single_video = determine_data_config(args.data, args.reprojection_name)

    # Create dataset and loader
    val_dataset, val_loader = create_dataset_and_loader(args, data_root, is_single_video, loop_args)

    # Setup model and pipeline
    pipeline, rays, weight_dtype = setup_model_and_pipeline(args)

    # Process batches
    for idx, batch in tqdm(enumerate(val_loader)):
        if idx < args.start_idx:
            continue
        if idx >= args.num_data + args.start_idx:
            break

        current_episode = val_dataset.episodes[idx]


        process_batch(
            batch,
            args,
            pipeline,
            rays,
            weight_dtype,
            output_path,
            current_episode,
            decode_chunk_size=args.decode_chunk_size,
        )



if __name__ == "__main__":
    main()
