import random

import numpy as np
import torch
from PIL import Image


def to_uint8_image_array(img):
    """
    Normalize an image-like array so PIL can consume it safely as uint8.

    Args:
        img: Array-like image data in any numeric dtype.

    Returns:
        np.ndarray: A uint8 image array clipped to [0, 255].
    """
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img

    if np.issubdtype(img.dtype, np.floating):
        max_value = float(np.nanmax(img)) if img.size else 0.0
        if max_value <= 1.0:
            img = img * 255.0

    return np.clip(img, 0, 255).astype(np.uint8)


def tensor_to_pil(x):
    """
    Convert a normalized CHW tensor into a PIL image.

    Args:
        x: torch.Tensor in CHW layout, assumed to be normalized to [-1, 1].

    Returns:
        PIL.Image.Image or the original input if `x` is not a tensor.
    """
    if not isinstance(x, torch.Tensor):
        return x
    # assume CHW in [-1, 1]
    x = (x * 0.5 + 0.5).clamp(0, 1)
    x = x.mul(255).byte().detach().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(x)


def frame_to_pil(frame):
    """
    Convert a frame to PIL if needed.

    Args:
        frame: Either a PIL image or a torch tensor in CHW layout.

    Returns:
        PIL.Image.Image: The input frame as a PIL image.
    """
    if isinstance(frame, Image.Image):
        return frame
    if isinstance(frame, torch.Tensor):
        return tensor_to_pil(frame)
    raise TypeError(f"Unsupported frame type: {type(frame)}")


def pil_to_tensor(img):
    """
    Convert a PIL image into a normalized CHW tensor.

    Args:
        img: PIL image in RGB format.

    Returns:
        torch.Tensor or the original input if `img` is not a PIL image.
    """
    if not isinstance(img, Image.Image):
        return img
    arr = np.asarray(img)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t * 2 - 1


def random_mask_frames(
    images,
    patch_size,
    patch_mask_ratio_min,
    patch_mask_ratio_max,
    pixel_mask_ratio_min,
    pixel_mask_ratio_max,
):
    """
    Apply hybrid random patch + random pixel masking to a clip of images.

    Frame 0 is kept clean; frames 1..N-1 are masked with two stages:
    1. Random patch masking
    2. Random pixel masking within the remaining unmasked regions

    Masked pixels are set to `-1.0`, which matches the dataset's normalized
    image range and the black-background convention used by reprojection.

    Args:
        images: torch.Tensor of shape [T, C, H, W] in the normalized range.
        patch_size: Size of each square patch used for patch masking.
        patch_mask_ratio_min: Lower bound for random patch masking ratio.
        patch_mask_ratio_max: Upper bound for random patch masking ratio.
        pixel_mask_ratio_min: Lower bound for random pixel masking ratio.
        pixel_mask_ratio_max: Upper bound for random pixel masking ratio.

    Returns:
        torch.Tensor: Masked clip with the same shape as `images`.
    """
    masked = images.clone()
    num_frames, channels, height, width = masked.shape

    num_patches_h = (height + patch_size - 1) // patch_size
    num_patches_w = (width + patch_size - 1) // patch_size
    total_patches = num_patches_h * num_patches_w

    for frame_idx in range(1, num_frames):
        patch_ratio = random.uniform(patch_mask_ratio_min, patch_mask_ratio_max)
        num_patches_to_mask = int(total_patches * patch_ratio)

        patch_indices = list(range(total_patches))
        random.shuffle(patch_indices)
        masked_patch_set = set(patch_indices[:num_patches_to_mask])

        patch_mask = torch.zeros(height, width, dtype=torch.bool)
        for patch_idx in masked_patch_set:
            patch_row = patch_idx // num_patches_w
            patch_col = patch_idx % num_patches_w
            h_start = patch_row * patch_size
            h_end = min(h_start + patch_size, height)
            w_start = patch_col * patch_size
            w_end = min(w_start + patch_size, width)
            patch_mask[h_start:h_end, w_start:w_end] = True

        pixel_ratio = random.uniform(pixel_mask_ratio_min, pixel_mask_ratio_max)
        pixel_rand = torch.rand(height, width)
        pixel_mask = (~patch_mask) & (pixel_rand < pixel_ratio)

        combined_mask = patch_mask | pixel_mask
        combined_mask_3c = combined_mask.unsqueeze(0).expand(channels, -1, -1)
        masked[frame_idx][combined_mask_3c] = -1.0

    return masked
