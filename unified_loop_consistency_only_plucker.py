#!/usr/bin/env python3


from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# -----------------------
# Env / sys path setup
# -----------------------
os.environ.setdefault("TORCH_HOME", "model_cache")
os.environ.setdefault("HF_HOME", "model_cache")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYGLET_HEADLESS", "1")
os.environ.setdefault("PYGLET_BACKEND", "egl")

sys.path.append("./evoworld")
sys.path.append("third_party/vggt/")

# -----------------------
# Project imports
# -----------------------
from evoworld.inference.navigator_evoworld import Navigator
from evoworld.inference.forward_evoworld_only_plucker import process_batch
from evoworld.trainer.unet_plucker import UNetSpatioTemporalConditionModel
from evoworld.pipeline.pipeline_evoworld_only_plucker import (
    StableVideoDiffusionOnlyPluckerPipeline as StableVideoDiffusionPipeline,
)
from conver_equi_cube import safe_equi2equi_resize
from utils.image_utils import frame_to_pil, pil_to_tensor, tensor_to_pil
from utils.plucker_embedding import equirectangular_to_ray

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("UnifiedRefactor")

# -----------------------
# Constants
# -----------------------
UNITY_TO_OPENCV = np.array([1, -1, 1, -1, 1, -1], dtype=float)
MULTI_SEGMENT_POS_SCALE = 0.1
DEFAULT_PANO_H, DEFAULT_PANO_W = 576, 1024
DEFAULT_PERS_H, DEFAULT_PERS_W = 384, 512

# -----------------------
# Small utilities
# -----------------------

def set_random_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_frames(frames: List[torch.Tensor], out_dir: str, start_idx: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, fr in enumerate(frames):
        p = os.path.join(out_dir, f"{i + start_idx + 1:03}.png")
        frame_to_pil(fr).save(p)


def save_comparison_frames(pred_frames, gt_frames, out_dir: str, start_idx: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if len(pred_frames) != len(gt_frames):
        raise ValueError(
            f"Prediction and GT frame counts must match, got {len(pred_frames)} and {len(gt_frames)}."
        )

    for i, (pred, gt) in enumerate(zip(pred_frames, gt_frames)):
        pred_pil = frame_to_pil(pred)
        gt_pil = frame_to_pil(gt)
        if gt_pil.size != pred_pil.size:
            gt_pil = gt_pil.resize(pred_pil.size, Image.BICUBIC)
        comparison = Image.new("RGB", (pred_pil.width, pred_pil.height + gt_pil.height))
        comparison.paste(pred_pil, (0, 0))
        comparison.paste(gt_pil, (0, pred_pil.height))
        p = os.path.join(out_dir, f"{i + start_idx + 1:03}.png")
        comparison.save(p)


# -----------------------
# Main pipeline
# -----------------------
class UnifiedLoopConsistencyPipeline:
    """Plucker-only inference pipeline aligned with empty_with_traj training."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Models (lazy init)
        self.navigator: Optional[Navigator] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing plucker-only pipeline...")

    # ---------- Model init ----------
    def initialize_models(self) -> None:
        self.logger.info("Loading Navigator (UNet + SVD pipeline)...")
        self.navigator = Navigator(
            height=self.args.height,
            width=self.args.width,
            decode_chunk_size=self.args.decode_chunk_size,
        )
        self.navigator.get_pipeline(
            self.args.unet_path,
            self.args.svd_path,
            pipeline_cls=StableVideoDiffusionPipeline,
            model_width=self.args.width,
            model_height=self.args.height,
            progress_bar=False,
            num_frames=self.args.num_frames,
        )
        self.logger.info("Navigator loaded. VGGT/Open3D memory conditioning is disabled.")

    def setup_model_and_pipeline(self, args: argparse.Namespace):
        """Setup UNet model and SVD pipeline for the single-segment fast path."""
        weight_dtype = torch.float32

        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            args.unet_path, subfolder="unet", low_cpu_mem_usage=True
        )
        unet.requires_grad_(False)

        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            args.unet_path, unet=unet, local_files_only=True, low_cpu_mem_usage=True
        )
        pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)

        rays = equirectangular_to_ray(target_H=args.height // 8, target_W=args.width // 8)
        rays = torch.tensor(rays, dtype=weight_dtype, device=self.device)

        return pipeline, rays, weight_dtype

    # ---------- Data init ----------
    def determine_data_config(self) -> Tuple[str, bool]:
        try:
            folder_contents = os.listdir(self.args.base_folder)
        except FileNotFoundError as e:
            raise ValueError(f"Base folder '{self.args.base_folder}' not found") from e
        is_single_video = "panorama" in folder_contents
        data_root = self.args.base_folder if is_single_video else os.path.join(self.args.base_folder, "test")
        return data_root, is_single_video

    def create_dataset_and_loader(self, data_root: str, is_single_video: bool):
        from dataset.CameraTrajDataset import CameraTrajDataset
        dataset = CameraTrajDataset(
            data_root,
            width=self.args.width,
            height=self.args.height,
            trajectory_file=None,
            memory_sampling_args={"sampling_method": "empty_with_traj", "include_initial_frame": True},
            load_complete_episode=False,
            reprojection_name="rendered_panorama_vggt_open3d",
            is_single_video=is_single_video,
            only_position=self.args.only_position,
            clip_start_frame=self.args.clip_start_frame,
        )

        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
        return dataset, loader

    # ---------- Episode helpers ----------
    def enumerate_episodes(self, data_root: str, is_single_video: bool) -> List[Tuple[str, str]]:
        if is_single_video:
            return [("", data_root)]

        episodes: List[Tuple[str, str]] = []
        for item in sorted(os.listdir(data_root)):
            episode_path = os.path.join(data_root, item)
            if os.path.isdir(episode_path) and "episode" in item:
                episodes.append((item, episode_path))
        return episodes

    def load_camera_poses(self, episode_path: str) -> np.ndarray:
        cam_file = os.path.join(episode_path, "camera_poses.txt")
        if not os.path.isfile(cam_file):
            raise FileNotFoundError(f"camera_poses.txt not found under {episode_path}")

        rows: List[List[float]] = []
        with open(cam_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if parts and ("Frame" in parts[0] or "frame" in parts[0]):
                    # header row
                    continue
                if len(parts) >= 7:
                    # skip frame_id, take next 6 values
                    vals = [float(x) for x in parts[1:7]]
                    rows.append(vals)
        if not rows:
            raise ValueError(f"No valid camera pose rows parsed from {cam_file}")

        cam = np.asarray(rows, dtype=float)
        # Unity->OpenCV convention flip
        cam *= UNITY_TO_OPENCV
        # Align direct pose loading with CameraTrajDataset's default pos_scale.
        cam[:, :3] *= MULTI_SEGMENT_POS_SCALE
        return cam

    def get_segment_bounds(self, segment_id: int, total_frames: int) -> Optional[Tuple[int, int]]:
        start_idx = segment_id * (self.args.num_frames - 1)
        if start_idx >= total_frames:
            return None

        end_idx = min(start_idx + self.args.num_frames, total_frames)
        if segment_id > 0 and end_idx - start_idx < 2:
            return None

        return start_idx, end_idx

    def resolve_frame_path(self, episode_path: str, frame_id: int) -> str:
        frame_path = os.path.join(episode_path, "panorama", f"{frame_id:03}.png")
        if os.path.isfile(frame_path):
            return frame_path
        raise FileNotFoundError(f"Panorama frame {frame_id:03} not found under {episode_path}/panorama")

    def load_frame_tensor(self, episode_path: str, frame_id: int) -> torch.Tensor:
        frame = Image.open(self.resolve_frame_path(episode_path, frame_id)).convert("RGB")
        frame_tensor = pil_to_tensor(frame)
        if frame_tensor.shape[-2:] != (self.args.height, self.args.width):
            frame_tensor = safe_equi2equi_resize(
                frame_tensor.unsqueeze(0),
                height=self.args.height,
                width=self.args.width,
                mode="bilinear",
            ).squeeze(0)
        return frame_tensor

    def load_gt_frames(self, episode_path: str, start_idx: int, end_idx: int) -> List[torch.Tensor]:
        frames: List[torch.Tensor] = []
        for frame_id in range(start_idx + 1, end_idx + 1):
            frames.append(self.load_frame_tensor(episode_path, frame_id))
        return frames

    def is_episode_complete(self, episode: str) -> bool:
        if not self.args.save_frames:
            return False

        episode_save_dir = os.path.join(self.args.save_dir, episode)
        if self.args.single_segment:
            predictions_dir = os.path.join(episode_save_dir, "predictions")
            if not os.path.isdir(predictions_dir):
                return False

            saved_frames = [
                name for name in os.listdir(predictions_dir)
                if name.lower().endswith(".png")
            ]
            return len(saved_frames) == self.args.num_frames

        for segment_id in range(self.args.num_segments):
            predictions_dir = os.path.join(episode_save_dir, f"predictions_{segment_id}")
            if not os.path.isdir(predictions_dir):
                return False

            expected_frames = self.args.num_frames if segment_id == 0 else self.args.num_frames - 1
            saved_frames = [
                name for name in os.listdir(predictions_dir)
                if name.lower().endswith(".png")
            ]
            if len(saved_frames) != expected_frames:
                return False

        return True

    @torch.inference_mode()
    def generate_segment(
        self,
        segment_traj: torch.Tensor,
        start_image: torch.Tensor,
    ) -> List[Image.Image]:
        assert self.navigator is not None

        navigate_fn = getattr(
            self.navigator,
            "navigate_curve_path" if self.args.curve_path else "navigate_path",
        )
        generations = navigate_fn(
            segment_traj,
            start_image,
            width=self.args.width,
            height=self.args.height,
            num_inference_steps=25,
            memorized_images=None,
            infer_segment=False,
            segment_id=1,
        )
        return [img for move in generations for img in move]

    # ---------- Episode orchestration ----------
    def process_episode(self, episode: str, episode_path: str) -> None:
        episode_label = episode or os.path.basename(os.path.normpath(episode_path))
        self.logger.info("Processing episode: %s", episode_label)

        episode_save_dir = os.path.join(self.args.save_dir, episode)
        os.makedirs(episode_save_dir, exist_ok=True)
        camera_params = self.load_camera_poses(episode_path)

        all_generated_frames: List[Image.Image] = []
        initial_frame = self.load_frame_tensor(episode_path, 1).to(self.device)

        for segment_id in range(self.args.num_segments):
            bounds = self.get_segment_bounds(segment_id, len(camera_params))
            if bounds is None:
                self.logger.info("Stopping at segment %s: no more frames available.", segment_id)
                break

            start_idx, end_idx = bounds
            segment_traj = torch.tensor(
                camera_params[start_idx:end_idx],
                dtype=torch.float32,
                device=self.device,
            )
            self.logger.info(
                "Processing segment %s with frame range [%s, %s).",
                segment_id,
                start_idx,
                end_idx,
            )

            if segment_id == 0:
                start_image = initial_frame
            else:
                start_image = pil_to_tensor(tensor_to_pil(all_generated_frames[-1])).to(self.device)

            generated_frames = self.generate_segment(segment_traj, start_image)

            if all_generated_frames:
                generated_frames = generated_frames[1:]  # avoid duplicating first frame
            all_generated_frames.extend(generated_frames)

            # Save predicted frames (optional)
            if self.args.save_frames:
                frames_path = os.path.join(episode_save_dir, f"predictions_{segment_id}")
                # Segment>0 drops the duplicated boundary frame, so shift filenames by +1.
                start_idx_seg = segment_id * (self.args.num_frames - 1)
                if segment_id > 0:
                    start_idx_seg += 1
                save_frames(generated_frames, frames_path, start_idx_seg)

                frames_gt_path = os.path.join(episode_save_dir, f"predictions_gt_{segment_id}")
                gt_frames = self.load_gt_frames(episode_path, start_idx, end_idx)
                if segment_id > 0:
                    gt_frames = gt_frames[1:]
                save_frames(gt_frames, frames_gt_path, start_idx_seg)

                compare_path = os.path.join(episode_save_dir, f"predictions_compare_{segment_id}")
                save_comparison_frames(generated_frames, gt_frames, compare_path, start_idx_seg)

        self.logger.info("Episode %s processing completed.", episode_label)

    @torch.inference_mode()
    def run_pipeline(self) -> None:
        if self.args.only_position:
            raise NotImplementedError(
                "--only_position is currently supported only in --single_segment mode."
            )

        set_random_seeds(self.args.seed)
        self.logger.info("Starting plucker-only unified pipeline...")
        self.logger.info("Mode: empty_with_traj + first frame + plucker embedding only")
        self.logger.info(f"UNet Path: {self.args.unet_path}")
        self.logger.info(f"SVD Path:  {self.args.svd_path}")

        self.initialize_models()
        data_root, is_single_video = self.determine_data_config()
        episodes = self.enumerate_episodes(data_root, is_single_video)

        for idx, (episode, episode_path) in tqdm(enumerate(episodes), total=len(episodes)):
            if idx < self.args.start_idx:
                continue
            if idx >= self.args.num_data + self.args.start_idx:
                break
            if self.args.skip_completed and self.is_episode_complete(episode):
                episode_label = episode or os.path.basename(os.path.normpath(episode_path))
                self.logger.info("Skipping completed episode: %s", episode_label)
                continue
            try:
                self.process_episode(episode, episode_path)
            finally:
                if self.navigator is not None:
                    self.navigator.clear_runtime_state()
                gc.collect()

    @torch.inference_mode()
    def run_single_segment(self) -> None:
        set_random_seeds(self.args.seed)
        self.logger.info("Starting plucker-only single-segment path...")
        self.logger.info("Mode: empty_with_traj + first frame + plucker embedding only")
        self.logger.info(f"UNet Path: {self.args.unet_path}")
        self.logger.info(f"SVD Path:  {self.args.svd_path}")

        pipeline, rays, weight_dtype = self.setup_model_and_pipeline(self.args)
        data_root, is_single_video = self.determine_data_config()
        val_dataset, _ = self.create_dataset_and_loader(data_root, is_single_video)

        for idx in tqdm(range(len(val_dataset)), total=len(val_dataset)):
            if idx < self.args.start_idx:
                continue
            if idx >= self.args.num_data + self.args.start_idx:
                break
            current_episode = val_dataset.episodes[idx]
            if self.args.skip_completed and self.is_episode_complete(current_episode):
                self.logger.info("Skipping completed episode: %s", current_episode)
                continue
            episode_save_dir = os.path.join(self.args.save_dir, current_episode)
            os.makedirs(episode_save_dir, exist_ok=True)
            batch = torch.utils.data.default_collate([val_dataset[idx]])
            # ensure arg for process_batch
            self.args.mask_mem = False
            self.logger.info("")
            process_batch(
                batch,
                self.args,
                pipeline,
                rays,
                weight_dtype,
                episode_save_dir,
                current_episode,
                decode_chunk_size=self.args.decode_chunk_size,
            )


# -----------------------
# CLI
# -----------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plucker-only unified inference pipeline")

    # Model paths
    parser.add_argument("--unet_path", type=str, required=True, help="Path to UNet model")
    parser.add_argument(
        "--svd_path",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        help="Path to SVD model",
    )

    # Data configuration
    parser.add_argument("--base_folder", type=str, default="data/Curve_Loop/test", help="Base folder containing episodes")
    parser.add_argument("--save_dir", type=str, default="unified_output", help="Output directory")
    parser.add_argument("--dataset_name", type=str, default="CameraTrajDataset", help="Dataset name")

    # Processing parameters
    parser.add_argument("--num_data", type=int, default=1, help="Number of episodes to process")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index")
    parser.add_argument("--num_segments", type=int, default=3, help="Number of segments to process")
    parser.add_argument("--num_frames", type=int, default=25, help="Frames per segment")
    parser.add_argument("--width", type=int, default=DEFAULT_PANO_W, help="Panorama width for dataset/model")
    parser.add_argument("--height", type=int, default=DEFAULT_PANO_H, help="Panorama height for dataset/model")
    parser.add_argument("--pers_width", type=int, default=DEFAULT_PERS_W, help="Perspective width for VGGT stage")
    parser.add_argument("--pers_height", type=int, default=DEFAULT_PERS_H, help="Perspective height for VGGT stage")
    parser.add_argument("--save_frames", action="store_true", help="Save intermediate frames")
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip episodes whose prediction frame counts are complete.",
    )
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=8,
        help="VAE decode chunk size. Smaller values reduce memory usage.",
    )
    parser.add_argument(
        "--clip_start_frame",
        type=int,
        default=None,
        help="Optional 1-based start frame id for single-segment evaluation. If omitted, the clip start is sampled.",
    )

    # Options
    parser.add_argument("--curve_path", action="store_true", help="Use curve path navigation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--single_segment", action="store_true", help="Use single segment fast path")
    parser.add_argument(
        "--only_position",
        action="store_true",
        help="Use aligned_to_first panoramas/poses and apply only-position evaluation logic.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    os.makedirs(args.save_dir, exist_ok=True)
    pipe = UnifiedLoopConsistencyPipeline(args)
    if args.single_segment:
        pipe.run_single_segment()
    else:
        pipe.run_pipeline()


if __name__ == "__main__":
    main()
