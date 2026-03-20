import json
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from .colorsetting import ColoredFormatter
except:
    from colorsetting import ColoredFormatter

import logging

from utils.constant import UNITY_TO_OPENCV

# Set up the logger
logger = logging.getLogger("colored_logger")
logger.setLevel(logging.INFO)
# Disable propagation to avoid duplicate logs
logger.propagate = False
# Create a stream handler
handler = logging.StreamHandler()
# Apply the colored formatter
colored_formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(colored_formatter)
# Add the handler to the logger
logger.handlers = []
logger.addHandler(handler)

all_close = lambda x, y: torch.allclose(x, y, atol=1e-2)


MEMORY_SAMPLING_ARGS = {
    "sampling_method": "reprojection",
    "include_initial_frame": True,
}


class CustomRescale:
    """
    class: rescale the pxiel from [0,1] to [-1,1]
    """

    def __init__(self):
        pass

    def __call__(self, img):
        return img * 2 - 1


def split_by_region(root, x_range, z_range):
    """
    split the dataset by the region
    Args:
        root: str, root directory of the dataset
        x_range: tuple, x range of the region
        z_range: tuple, z range of the region
    Returns:
        split:  dict, a dictionary containing the following
            keys:  data_split(train,test)
            values: list of episode names
    """
    split = {"train": [], "test": []}
    for item in os.listdir(root):
        path = os.path.join(root, item)
        if os.path.isdir(path) and "episode" in item:
            episode = item.split("_")[-1]
            camera_pose = load_camera_poses_from_txt(
                os.path.join(root, item, "camera_poses.txt")
            )
            x_max, x_min, z_max, z_min = get_max_min(camera_pose)
            if in_box(x_max, x_min, z_max, z_min, x_range, z_range):
                split["test"].append(item)
            else:
                split["train"].append(item)
    return split


def get_max_min(camera_poses):
    """
    get the max and min of the camera poses
    Args:
        camera_poses: dict, a dictionary containing the following
            keys:  frame_id
            values: list of camera poses
    Returns:
        x_max: float, max x
        x_min: float, min x
        z_max: float, max z
        z_min: float, min z
    """
    x_max = -float("inf")
    x_min = float("inf")
    z_max = -float("inf")
    z_min = float("inf")
    for pose in camera_poses.values():
        x, y, z, _, _, _ = pose
        x_max = max(x_max, x)
        x_min = min(x_min, x)
        z_max = max(z_max, z)
        z_min = min(z_min, z)
    return x_max, x_min, z_max, z_min


def in_box(x_max, x_min, z_max, z_min, x_range, z_range):
    """
    check if two boxes intersect
    Args:
        x_max: float, max x of first box
        x_min: float, min x of first box
        z_max: float, max z of first box
        z_min: float, min z of first box
        x_range: tuple, x range of the second box
        z_range: tuple, z range of the second box
    Returns:
        bool: True if in the box, False otherwise
    """
    x2_min, x2_max = x_range
    z2_min, z2_max = z_range

    # Check if there is overlap in both x and z directions
    return not (x_max < x2_min or x_min > x2_max or z_max < z2_min or z_min > z2_max)


def load_camera_poses_from_txt(file_path):
    """
    Load camera poses from a file into a dictionary.

    Args:
        file_path (str): Path to the input file containing camera poses.

    Returns:
        dict: A dictionary where the key is frame_id (int) and the value is a list [x, y, z, rotx, roty, rotz].
    """
    camera_poses = {}

    with open(file_path, "r") as file:
        # Skip the header line
        lines = file.readlines()[1:]

        for line in lines:
            # Split the line into individual values
            values = line.strip().split(",")

            # Extract frame_id, position, and rotation values
            frame_id = values[0]
            x, y, z = map(float, values[1:4])
            rotx, roty, rotz = map(float, values[4:7])

            # Store as a list [x, y, z, rotx, roty, rotz]
            camera_poses[frame_id] = [x, y, z, rotx, roty, rotz]

    return camera_poses


def build_traj_file_from_raw_info(root, episodes):
    """
    Load camera trajectories from a JSON file.

    Args:
        root: str, root directory of the dataset
        episodes: list, list of episode names

    Returns:
        dict: A two-layered dictionary where the key is episode name (str) and frame_id, the value is a dictionary of camera poses.
    """
    traj_file_path = os.path.join(root, "camera_trajectories.json")

    # Fast path: load from cache if it already exists and covers all episodes
    if os.path.exists(traj_file_path):
        with open(traj_file_path, "r") as f:
            cached = json.load(f)
        if all(ep in cached for ep in episodes):
            logger.info(
                f"Loaded cached camera_trajectories.json from {traj_file_path} "
                f"({len(episodes)} episodes)."
            )
            return cached
        logger.info(
            "Cached camera_trajectories.json is incomplete; rebuilding..."
        )

    # Slow path: read every camera_poses.txt and write the cache
    camera_poses = {}
    for episode in episodes:
        camera_poses[episode] = load_camera_poses_from_txt(
            os.path.join(root, episode, "camera_poses.txt")
        )

    with open(traj_file_path, "w") as traj_file:
        json.dump(camera_poses, traj_file, indent=4)

    return camera_poses


def load_trajectory_file(traj_file):
    """
    Load camera trajectories from a JSON file.

    Args:
        traj_file (str): Path to the input JSON file containing camera trajectories.

    Returns:
        dict: A two-layered dictionary where the key is episode name (str) and frame_id, the value is a dictionary of camera poses.
    """
    with open(traj_file, "r") as file:
        camera_trajectories = json.load(file)

    return camera_trajectories


class CameraTrajDataset(Dataset):
    """
    CameraTrajDataset is a class that loads

        panoramic images,
        camera poses and
        memories[images, camera poses]

    from specified data dir, camera trajectories info (if given) and other needed args.
    """

    def __init__(
        self,
        root: str,
        height: int = 1000,
        width: int = 2000,
        transform=CustomRescale(),
        trajectory_file: str = None,
        sequence_length: int = 25,
        memory_sampling_args: dict = MEMORY_SAMPLING_ARGS,
        no_images: bool = False,
        last_segment_length: int = 25,
        pos_scale: float = 0.1,
        data_source: str = "unity",
        image_name_prefix: str = "",
        load_complete_episode=False,
        id_zero_start=False,
        memory_path: str = None,
        reprojection_name="rendered_panorama",
        is_single_video=False,
        pretrain_mode=False,
        mask_ratio_min=0.3,
        mask_ratio_max=0.9,
        stride_min=1,
        stride_max=10,
        patch_size=32,
        patch_mask_ratio_min=0.2,
        patch_mask_ratio_max=0.6,
        pixel_mask_ratio_min=0.1,
        pixel_mask_ratio_max=0.4,
    ):
        """
        Args:
            root: str, root directory of the dataset
            height: int, height of the image
            width: int, width of the image
            transform: callable, transform to apply to the image
            trajectory_file: str, json file that contains camera trajectories
            sequence_length: int, length of the training sequence
            no_images: bool, if True, only load camera trajectories,
            data_source: str, data source of the dataset
        """
        self.root = root
        self.sequence_length = sequence_length
        self.memory_sampling_args = memory_sampling_args
        self.image_name_prefix = image_name_prefix
        self.load_complete_episode = load_complete_episode
        self.reprojection_name = reprojection_name
        self.memory_path = memory_path
        self.pretrain_mode = pretrain_mode
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.stride_min = stride_min
        self.stride_max = stride_max
        self.patch_size = patch_size
        self.patch_mask_ratio_min = patch_mask_ratio_min
        self.patch_mask_ratio_max = patch_mask_ratio_max
        self.pixel_mask_ratio_min = pixel_mask_ratio_min
        self.pixel_mask_ratio_max = pixel_mask_ratio_max

        # detect all episodes
        self.episodes = []
        if is_single_video:
            self.episodes.append('')
        else:
            sorted_path_list = sorted(os.listdir(self.root))
            for item in sorted_path_list:
                path = os.path.join(self.root, item)
                if os.path.isdir(path) and "episode" in item:
                    self.episodes.append(item)

        # image format
        self.height = height
        self.width = width
        self.last_segment_length = last_segment_length

        if trajectory_file is None:
            trajectories = build_traj_file_from_raw_info(self.root, self.episodes)
        else:
            trajectories = load_trajectory_file(trajectory_file)

        # convert camera poses to OpenCV coordinate system
        trajectories = self.convert_to_opencv_rdf(trajectories)

        # load camera trajectories by aggregating all episodes
        self.trajectories = self.build_traj_map(trajectories)

        self.transform = self.build_transform_pipeline(height, width, transform)

        self.current_trajectory = None
        self.current_memory = None
        self.no_images = no_images
        self.pos_scale = pos_scale

        self.id_zero_start = id_zero_start

        logger.info(f"CameraTrajDataset loaded with {len(self.episodes)} episodes.")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        """
        Args:
            idx: int, index of the episode
        Returns:
            dict: a dictionary containing the following keys:
                'pixel_values': torch.Tensor, [Seq C H W]
                'cam_traj': torch.Tensor, [Seq 6] (6: x, y, z, rotx(along x) roty(along y) rotz(along z))
                'memorized_pixel_values': torch.Tensor, [Num_M C H W]
                'memorized_cam_traj': torch.Tensor, [Num_M 6]
        """
        # get current episode
        # print("Episode: ",self.episodes[idx])
        # print("idx: ",idx)
        current_episode = self.episodes[idx]
        # current_episode = 'episode_7117'
        # print("current_episode: ",current_episode)
        #  get current episode length
        episode_length = len(
            self.trajectories["raw_trajectories"][current_episode].keys()
        )
        first_frame_id = 0 if self.id_zero_start else 1
        sampled_initial_frame_idx = first_frame_id
        initial_frame_traj = None
        initial_frame_image = None

        if self.pretrain_mode:
            # Pretrain mode: random start + random stride sampling
            frame_ids = self._sample_frames_with_stride(episode_length)
            current_images = self._load_images_by_ids(current_episode, frame_ids)
            current_traj = self._load_traj_by_ids(current_episode, frame_ids)
            # Generate masked GT frames as memorized_pixel_values
            memorized_images = self._random_mask_frames(current_images)
            start_idx = frame_ids[0]
            sampled_initial_frame_idx = frame_ids[0]
        else:
            if self.memory_sampling_args.get("sampling_method") == "empty_with_traj":
                # Sample a random contiguous clip without using the episode's first
                # frame as the start frame when possible.
                frame_ids = self._sample_contiguous_frames(
                    episode_length, exclude_episode_first_frame=True
                )
                current_images = self._load_images_by_ids(current_episode, frame_ids)
                current_traj = self._load_traj_by_ids(current_episode, frame_ids)
                start_idx = frame_ids[0]
                sampled_initial_frame_idx = frame_ids[0]
            else:
                # Original mode: fixed segment + reprojection
                valid_range_start_idx = (
                    episode_length - self.last_segment_length + 1
                    if not self.load_complete_episode
                    else 1
                )
                valid_range_start_idx = (
                    valid_range_start_idx - 1
                    if self.id_zero_start
                    else valid_range_start_idx
                )
                # Given the sequence length, retrieve a fixed sequence of frames from the episode
                start_idx = valid_range_start_idx
                end_idx = (
                    start_idx + self.sequence_length
                    if not self.load_complete_episode
                    else start_idx + episode_length
                )
                # load the images: [Seq C H W]
                current_images = self.load_images(current_episode, start_idx, end_idx)
                # load the camera trajectory: [Seq 6]: [x, y, z, rotx(along x) roty(along y) rotz(along z)]
                current_traj = self.load_traj(current_episode, start_idx, end_idx)

        # store the current trajectory for display
        self.current_trajectory = current_traj.clone()
        self.current_episode_trajectory = torch.tensor(
            [
                pose
                for pose in self.trajectories["raw_trajectories"][
                    current_episode
                ].values()
            ]
        )

        if not self.pretrain_mode:
            # sample associated memories(images and trajectories) from pre-built trajectory map
            memories = self.sampling_memories(current_traj, current_episode, start_idx)
            memorized_images = memories["images"]

        current_traj[:, :3] = current_traj[:, :3] * self.pos_scale

        # store the current memory for display
        self.current_memory = dict()
        if self.pretrain_mode:
            self.current_memory["traj"] = current_traj.clone()
            self.current_memory["images"] = memorized_images.clone()
            memorized_traj = current_traj.clone()
        else:
            self.current_memory["traj"] = memories["traj"].clone()
            self.current_memory["images"] = memories["images"].clone()
            memories["traj"][:, :3] = memories["traj"][:, :3] * self.pos_scale
            memorized_traj = memories["traj"]

        if self.memory_sampling_args.get("include_initial_frame", False):
            initial_frame_traj = self.load_traj(
                current_episode, sampled_initial_frame_idx, sampled_initial_frame_idx + 1
            )
            initial_frame_traj[:, :3] = initial_frame_traj[:, :3] * self.pos_scale
            initial_frame_image = self.load_images(
                current_episode, sampled_initial_frame_idx, sampled_initial_frame_idx + 1
            )
        # print("loaded episode: ",current_episode)
        return {
            "pixel_values": current_images,
            "cam_traj": current_traj,
            "memorized_pixel_values": memorized_images,
            "memorized_cam_traj": memorized_traj,
            "initial_frame_traj": initial_frame_traj,
            "initial_frame_image": initial_frame_image,
            "episode_path": os.path.join(self.root, current_episode),
        }

    # ==================== Pretrain mode helpers ====================

    def _sample_frames_with_stride(self, episode_length):
        """
        Randomly sample frame indices with a random stride.
        1. Random start frame
        2. Random stride (clamped so that all num_frames fit within episode)
        Args:
            episode_length: int, total number of frames in the episode
        Returns:
            list[int]: list of frame indices (1-based, matching trajectory keys)
        """
        first_frame_id = 0 if self.id_zero_start else 1
        last_frame_id = (episode_length - 1) if self.id_zero_start else episode_length
        num_frames = self.sequence_length

        # Random start
        # max_start ensures start + stride * (num_frames - 1) <= last_frame_id with stride=1
        max_start = last_frame_id - (num_frames - 1)
        start = random.randint(first_frame_id, max(first_frame_id, max_start))

        # Random stride, clamped to fit
        remaining = last_frame_id - start
        max_possible_stride = remaining // (num_frames - 1) if num_frames > 1 else 1
        stride_lo = min(self.stride_min, max_possible_stride)
        stride_hi = min(self.stride_max, max_possible_stride)
        stride = random.randint(max(1, stride_lo), max(1, stride_hi))

        frame_ids = [start + i * stride for i in range(num_frames)]
        return frame_ids

    def _sample_contiguous_frames(
        self, episode_length, exclude_episode_first_frame=False
    ):
        """
        Randomly sample a contiguous clip with stride 1.

        Args:
            episode_length: int, total number of frames in the episode
            exclude_episode_first_frame: bool, avoid using the episode's first
                frame as the sampled start frame when the episode is long enough.
        Returns:
            list[int]: list of contiguous frame indices
        """
        first_frame_id = 0 if self.id_zero_start else 1
        last_frame_id = (episode_length - 1) if self.id_zero_start else episode_length
        num_frames = self.sequence_length

        max_start = last_frame_id - (num_frames - 1)
        if max_start < first_frame_id:
            raise ValueError(
                f"Episode length {episode_length} is shorter than sequence length {num_frames}."
            )

        min_start = (
            first_frame_id + 1
            if exclude_episode_first_frame and max_start >= first_frame_id + 1
            else first_frame_id
        )
        start = random.randint(min_start, max_start)

        return [start + i for i in range(num_frames)]

    def _load_images_by_ids(self, episode, frame_ids):
        """
        Load images by a list of frame indices.
        Args:
            episode: str, episode name
            frame_ids: list[int], frame indices
        Returns:
            torch.Tensor: [Seq C H W]
        """
        if self.no_images:
            return torch.zeros(len(frame_ids), 3, self.height, self.width)
        images = []
        for fid in frame_ids:
            image_name = self.image_name_prefix + f"{fid:03}.png"
            images_path = os.path.join(self.root, episode, "panorama", image_name)
            if not os.path.exists(images_path):
                image_name = self.image_name_prefix + f"{fid:03}.jpg"
                images_path = os.path.join(self.root, episode, "panorama", image_name)
            cur_images = Image.open(images_path).convert("RGB")
            cur_images = self.transform(cur_images)
            images.append(cur_images)
        return torch.stack(images)

    def _load_traj_by_ids(self, episode, frame_ids):
        """
        Load camera trajectory by a list of frame indices.
        Args:
            episode: str, episode name
            frame_ids: list[int], frame indices
        Returns:
            torch.Tensor: [Seq 6]
        """
        traj = []
        for fid in frame_ids:
            traj.append(self.trajectories["raw_trajectories"][episode][str(fid)])
        return torch.tensor(traj)

    def _random_mask_frames(self, images):
        """
        Apply hybrid Random Patch + Random Pixel masking to GT frames for pretrain mode.
        Frame 0 (first frame) is kept clean; frames 1..N-1 are masked in two stages:
          Stage 1 - Patch masking: divide the image into patch_size × patch_size patches,
                    randomly mask a proportion of patches (whole patch set to -1).
          Stage 2 - Pixel masking: in the remaining *unmasked* patches, randomly mask
                    individual pixels (set to -1).
        Masked pixels are set to -1 (black in [-1,1] range, i.e. 0 in [0,1]).
        This matches the EvoWorld VGGT reprojection convention where Open3D renders
        empty/background regions as black (RGB 0,0,0), which after ToTensor + CustomRescale
        becomes -1 in [-1,1] range.

        Total effective mask ratio ≈ patch_ratio + (1 - patch_ratio) × pixel_ratio

        Args:
            images: torch.Tensor, [Seq C H W], pixel range [-1, 1]
        Returns:
            torch.Tensor: [Seq C H W], masked frames
        """
        masked = images.clone()
        num_frames, C, H, W = masked.shape
        ps = self.patch_size

        # Number of patches along each dimension (ceiling division handles remainder)
        num_patches_h = (H + ps - 1) // ps
        num_patches_w = (W + ps - 1) // ps
        total_patches = num_patches_h * num_patches_w

        for i in range(1, num_frames):  # skip first frame
            # ---- Stage 1: Random Patch Masking ----
            patch_ratio = random.uniform(self.patch_mask_ratio_min, self.patch_mask_ratio_max)
            num_patches_to_mask = int(total_patches * patch_ratio)

            # Randomly select which patches to mask
            patch_indices = list(range(total_patches))
            random.shuffle(patch_indices)
            masked_patch_set = set(patch_indices[:num_patches_to_mask])

            # Build a pixel-level mask from selected patches: True = masked by patch
            patch_mask = torch.zeros(H, W, dtype=torch.bool)
            for p_idx in masked_patch_set:
                ph = p_idx // num_patches_w
                pw = p_idx % num_patches_w
                h_start = ph * ps
                h_end = min(h_start + ps, H)
                w_start = pw * ps
                w_end = min(w_start + ps, W)
                patch_mask[h_start:h_end, w_start:w_end] = True

            # ---- Stage 2: Random Pixel Masking (on unmasked region) ----
            pixel_ratio = random.uniform(self.pixel_mask_ratio_min, self.pixel_mask_ratio_max)
            # Generate random pixel mask for the full image
            pixel_rand = torch.rand(H, W)
            # Only apply pixel mask where patch_mask is False (unmasked patches)
            pixel_mask = (~patch_mask) & (pixel_rand < pixel_ratio)

            # ---- Combine both masks ----
            combined_mask = patch_mask | pixel_mask  # [H, W]
            combined_mask_3c = combined_mask.unsqueeze(0).expand(C, -1, -1)  # [C, H, W]
            masked[i][combined_mask_3c] = -1.0  # black in [-1,1], matches reprojection background

        return masked

    # ==================== End pretrain mode helpers ====================

    def convert_to_opencv_rdf(self, trajectories):
        """
        convert camera poses of different coordinate system to OpenCV coordinate system.
        Args:
            trajectories: dict, a two-layered dictionary where the key is episode name (str) and frame_id, the value is a dictionary of camera poses.
        Returns:
            dict: a two-layered dictionary where the key is episode name (str) and frame_id, the value is a dictionary of camera poses.
        """
        for episode, episode_value in trajectories.items():
            for frame_id, pose in episode_value.items():
                x, y, z, rotx, roty, rotz = pose
                x, y, z = (
                    x * UNITY_TO_OPENCV[0],
                    y * UNITY_TO_OPENCV[1],
                    z * UNITY_TO_OPENCV[2],
                )
                rotx, roty, rotz = (
                    rotx * UNITY_TO_OPENCV[3],
                    roty * UNITY_TO_OPENCV[4],
                    rotz * UNITY_TO_OPENCV[5],
                )
                episode_value[frame_id] = [x, y, z, rotx, roty, rotz]
        return trajectories

    def build_traj_map(self, trajectories: dict):
        """
        build a trajectory map by aggregating raw trajectories.
        Args:
            trajectories: dict, a two-layered dictionary where the key is episode name (str) and frame_id, the value is a dictionary of camera poses.
        Returns:
            dict: a dictionary containing the following keys:
                all_poses: torch.tensor, list of all camera poses
                all_metadata: list, list of all metadata
        """
        all_poses = []
        all_metadata = []
        for episode_id, episode_value in trajectories.items():
            for frame_id, pose in episode_value.items():
                x, y, z, rotx, roty, rotz = pose
                all_poses.append([x, y, z, rotx, roty, rotz])
                all_metadata.append({"episode_id": episode_id, "frame_id": frame_id})
        all_poses = torch.tensor(all_poses)
        return {
            "raw_trajectories": trajectories,
            "all_poses": all_poses,
            "all_metadata": all_metadata,
        }

    def load_images(self, episode, start_idx, end_idx):
        """
        load image from episode and frame index
        Args:
            episode: str, episode name
            start_idx: int, start index of the sequence
            end_idx: int, end index of the sequence
        Returns:
            torch.Tensor: [Seq C H W]
        """
        if self.no_images:
            images = torch.zeros(end_idx - start_idx, 3, self.height, self.width)
            return images
        images = []
        #  load image from episode
        for i in range(start_idx, end_idx):
            image_name = self.image_name_prefix + f"{i:03}.png"
            images_path = os.path.join(self.root, episode, "panorama", image_name)
            if not os.path.exists(images_path):
                print(f"File '{images_path}' not found, try find jpg.")
                image_name = self.image_name_prefix + f"{i:03}.jpg"
                images_path = os.path.join(self.root, episode, "panorama", image_name)
            cur_images = Image.open(images_path)
            cur_images = cur_images.convert("RGB")
            cur_images = self.transform(cur_images)
            images.append(cur_images)

        return torch.stack(images) if len(images) > 1 else images[0]

    def load_reprojection(self, episode):
        """
        load re-projected   from episode and frame index
        Args:
            episode: str, episode name
            start_idx: int, start index of the sequence
            end_idx: int, end index of the sequence
        Returns:
            torch.Tensor: [Seq C H W]
        """
        if self.no_images:
            images = torch.zeros(self.sequence_length, 3, self.height, self.width)
            return images
        images = []
        if self.memory_path is None:
            current_reprojection_path = os.path.join(
                self.root, episode, self.reprojection_name
            )
        else:
            current_reprojection_path = os.path.join(
                self.memory_path, episode, self.reprojection_name
            )
        reprojection_list = sorted(os.listdir(current_reprojection_path))
        # only keep *.png files
        reprojection_list = [f for f in reprojection_list if f.endswith(".png")]
        reprojection_length = len(reprojection_list)
        # 限制加载的重投影图像数量，使其与 last_segment_length - 1 匹配
        # 因为后面会添加 first_frame，所以这里加载 last_segment_length - 1 张
        max_reprojection_frames = self.last_segment_length - 1
        reprojection_length = min(reprojection_length, max_reprojection_frames)
        #  load image from episode
        for i in range(0, reprojection_length):
            image_name = self.image_name_prefix + f"{i:02}.png"
            if self.memory_path is None:
                images_path = os.path.join(
                    self.root, episode, self.reprojection_name, image_name
                )
            else:
                images_path = os.path.join(
                    self.memory_path, episode, self.reprojection_name, image_name
                )
            if not os.path.exists(images_path):
                print(f"File '{images_path}' not found, try find jpg.")
                image_name = self.image_name_prefix + f"{i:02}.jpg"
                if self.memory_path is None:
                    images_path = os.path.join(
                        self.root, episode, self.reprojection_name, image_name
                    )
                else:
                    images_path = os.path.join(
                        self.memory_path, episode, self.reprojection_name, image_name
                    )
            cur_images = Image.open(images_path)
            cur_images = cur_images.convert("RGB")
            cur_images = self.transform(cur_images)
            images.append(cur_images)
        first_frame = self.load_images(episode, 1, 2)
        # TODO: consider only loading the first frame without transformation, since the reprojection images are already transformed. This can avoid potential inconsistency between the first frame and reprojection images.
        # # consider changing to:
        # episode_length = len(self.trajectories['raw_trajectories'][episode].keys())
        # valid_range_start_idx = episode_length - self.last_segment_length + 1 if not self.load_complete_episode else 1
        # valid_range_start_idx = valid_range_start_idx - 1 if self.id_zero_start else valid_range_start_idx
        # first_frame = self.load_images(episode, valid_range_start_idx, valid_range_start_idx+1)
        images.insert(0, first_frame)

        return torch.stack(images) if len(images) > 1 else images[0]

    def load_traj(self, episode, start_idx, end_idx):
        """
        load camera trajectory from episode and frame index
        Args:
            episode: str, episode name
            start_idx: int, start index of the sequence
            end_idx: int, end index of the sequence
        Returns:
            torch.Tensor: [Seq 6]
        """
        traj = []
        for i in range(start_idx, end_idx):
            i = str(i)
            traj.append(self.trajectories["raw_trajectories"][episode][i])

        return torch.tensor(traj)

    def sampling_memories(self, current_traj, current_episode, start_idx):
        """
        sample memories from the trajectory map.
        Args:
            current_traj: torch.Tensor, [Seq 6]
            current_episode: str, current episode name
        Returns:
            dict: a dictionary containing the following keys:
                'images': torch.Tensor, [Num_M C H W]
                'traj': torch.Tensor, [Num_M 6]
        """
        if self.memory_sampling_args["sampling_method"] == "reprojection":
            return self.get_reprojection_memory(current_episode, current_traj)
        elif self.memory_sampling_args["sampling_method"] == "empty_with_traj":
            return self.get_empty_memory_with_trajectories(
                current_episode, current_traj
            )
        else:
            raise ValueError(
                f"Sampling method '{self.memory_sampling_args['sampling_method']}' not supported."
            )

    def get_empty_memory_with_trajectories(self, current_episode, current_traj):
        """
        Get empty memory (zeros) in the same shape as reprojection memory.
        Args:
            current_episode: str, episode name (unused)
            current_traj: torch.Tensor, [Seq, 6]
        Returns:
            dict: a dictionary containing the following keys:
                'images': torch.Tensor, [Seq, C, H, W]
                'traj': torch.Tensor, [Seq, 6]
        """
        num_frames = current_traj.shape[0]
        return {
            "images": torch.zeros(num_frames, 3, self.height, self.width),
            "traj": current_traj,
        }

    def get_reprojection_memory(self, current_episode, current_traj):
        """
        get reprojection from rendered panorama
        Args:
            current_episode: str, current episode name
            episode: str, episode name
            current_traj: torch.Tensor, [Seq 6]
        Returns:
            dict: a dictionary containing the following keys:
                'images': torch.Tensor, [Num_M C H W]
                'traj': torch.Tensor, [Num_M 6]
        """
        #  load image from episode
        images = self.load_reprojection(current_episode)
        # load the camera trajectory: [Seq 6]: [x, y, z, rotx(along x) roty(along y) rotz(along z)]
        traj = current_traj.clone()
        return {"images": images, "traj": traj}

    def build_transform_pipeline(self, height, width, transform=None):
        """
        Args:
            height: int, height of the image
            width: int, width of the image
            transform: callable, custom transform to apply to the image.
                    Defaults to None.
        Returns:
            callable: transform pipeline.
        """
        # Define the default transform pipeline
        default_transform_pipeline = [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ]

        # Create the complete transform pipeline
        transform_pipeline = default_transform_pipeline

        # If a custom transform is provided, append it to the pipeline
        if transform is not None:
            if isinstance(transform, list):
                transform_pipeline.extend(
                    transform
                )  # Extend if it's a list of transforms
            else:
                transform_pipeline.append(
                    transform
                )  # Append if it's a single transform

        # Compose the pipeline
        transform_pipeline = transforms.Compose(transform_pipeline)

        return transform_pipeline


def custom_collate_fn(batch):
    """
    convert a list of samples to a batch
    """
    batched_pixel_values = torch.stack([sample["pixel_values"] for sample in batch])
    batched_cam_traj = torch.stack([sample["cam_traj"] for sample in batch])
    batched_memorized_pixel_values = torch.stack(
        [sample["memorized_pixel_values"] for sample in batch]
    )
    batched_memorized_cam_traj = torch.stack(
        [sample["memorized_cam_traj"] for sample in batch]
    )

    return {
        "pixel_values": batched_pixel_values,
        "cam_traj": batched_cam_traj,
        "memorized_pixel_values": batched_memorized_pixel_values,
        "memorized_cam_traj": batched_memorized_cam_traj,
    }


def xyz_euler_to_three_by_four_matrix_batch(
    xyz_euler, relative=False, flatten=False, debug=False, euler_as_rotation=False
):
    """
    Convert a batch of xyz euler angles to 3x4 transformation matrices.

    Args:
        xyz_euler (torch.Tensor): [B, 6], where each row is
            [x, y, z, rotx, roty, rotz].
        relative (bool): If True, make all frames relative to the first frame.
        flatten (bool): If True, return [B, 12] instead of [B, 3, 4].
        debug(bool): If True, return first frame repeat B times.

    Returns:
        torch.Tensor:
            [B, 3, 4] if flatten=False, else [B, 12].
    """
    batch_size = xyz_euler.size(0)

    # Split out components
    x, y, z, rotx, roty, rotz = torch.split(xyz_euler, 1, dim=1)

    rotx, roty, rotz = (
        rotx * torch.pi / 180,
        roty * torch.pi / 180,
        rotz * torch.pi / 180,
    )

    zero = torch.zeros_like(x)
    one = torch.ones_like(x)

    # Build rotation matrices for each Euler angle
    # Rx
    Rx = torch.cat(
        [
            one,
            zero,
            zero,
            zero,
            torch.cos(rotx),
            -torch.sin(rotx),
            zero,
            torch.sin(rotx),
            torch.cos(rotx),
        ],
        dim=1,
    ).view(batch_size, 3, 3)

    # Ry
    Ry = torch.cat(
        [
            torch.cos(roty),
            zero,
            torch.sin(roty),
            zero,
            one,
            zero,
            -torch.sin(roty),
            zero,
            torch.cos(roty),
        ],
        dim=1,
    ).view(batch_size, 3, 3)

    # Rz
    Rz = torch.cat(
        [
            torch.cos(rotz),
            -torch.sin(rotz),
            zero,
            torch.sin(rotz),
            torch.cos(rotz),
            zero,
            zero,
            zero,
            one,
        ],
        dim=1,
    ).view(batch_size, 3, 3)

    # Combined rotation R = Rz * Ry * Rx
    R = torch.bmm(Rz, torch.bmm(Ry, Rx))

    # Translation
    T = torch.cat([x, y, z], dim=1).view(batch_size, 3, 1)

    # Combine into [3 x 4]
    F = torch.cat([R, T], dim=2)  # [B, 3, 4]

    if relative:
        # -- Make everything relative to the first frame. --
        # We want:  F_rel[i] = F[0]^{-1} * F[i].
        # A 3x4 transform can be extended to a 4x4 by adding [0 0 0 1].
        # Or we can do it analytically (since  F[0]^{-1} = [R0^T, -R0^T t0]).

        # 1) Extract R0, t0 for the first frame
        R0 = F[0, :, :3].unsqueeze(0)  # [1, 3, 3]
        t0 = F[0, :, 3:].unsqueeze(0)  # [1, 3, 1]

        # 2) Compute R0^T once (inverse of rotation is transpose,
        #    because R0 is orthonormal)
        R0_inv = R0.transpose(1, 2)  # [1, 3, 3]

        # 3) For each frame i,
        #    R_rel[i] = R0_inv * R[i],
        #    t_rel[i] = R0_inv * (t[i] - t0).

        # Get R[i], t[i] from F for all i
        R_all = F[:, :, :3]  # [B, 3, 3]
        t_all = F[:, :, 3:]  # [B, 3, 1]

        # Expand R0_inv to [B, 3, 3] for bmm
        R0_inv_expanded = R0_inv.expand(batch_size, -1, -1)  # [B, 3, 3]
        # Subtraction broadcast: (t_all - t0) => [B, 3, 1]
        # Then multiply R0_inv * (t_all - t0)
        R_rel = torch.bmm(R0_inv_expanded, R_all)  # [B, 3, 3]
        t_rel = torch.bmm(R0_inv_expanded, (t_all - t0))  # [B, 3, 1]

        # Re-assemble [3 x 4] for each frame
        F_rel = torch.cat([R_rel, t_rel], dim=2)  # [B, 3, 4]
        F = F_rel

    if debug:
        F = F[0].repeat(batch_size, 1, 1)

    if flatten:
        F = F.view(batch_size, 12)

    if euler_as_rotation:
        if relative:
            rotx = rotx - rotx[0]
            roty = roty - roty[0]
            rotz = rotz - rotz[0]
        translation_part = F[:, :, 3]
        F = torch.cat([translation_part, rotx, roty, rotz], dim=1)

    return F


def inverse_transform(img):
    """
    Args:
        img: torch.Tensor, [C, H, W]
    Returns:
        torch.Tensor, [C, H, W]
    """
    img = img * 0.5 + 0.5
    img = img * 255
    img = img.to(torch.uint8)
    img = img.permute(1, 2, 0)
    img = img.cpu().numpy()
    return img


def save_image(img, path):
    """
    Args:
        img: torch.Tensor, [C, H, W]
    """
    img = inverse_transform(img)
    img = Image.fromarray(img)
    img.save(path)
