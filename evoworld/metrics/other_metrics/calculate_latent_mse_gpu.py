import os

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm


def load_inception_v4_model(device):
    from timm import create_model

    model = create_model("inception_v4", pretrained=True, num_classes=0)
    model.eval()
    model.to(device)
    print("Loaded Inception Model.")
    return model


def default_transform():
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_latent_batch_size():
    return int(os.environ.get("LATENT_BATCH_SIZE", "32"))


def calculate_latent_mse(image1_tensor, image2_tensor, model=None, transform=None):
    """Calculate latent MSE with batched GPU Inception-v4 inference."""
    print("calculate_latent_mse_gpu...")

    batch_size, n_frames, c, h, w = image1_tensor.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_batch_size = get_latent_batch_size()

    if model is None:
        model = load_inception_v4_model(device)
    else:
        model.eval()
        model.to(device)

    if transform is None:
        transform = default_transform()

    image1_tensor = image1_tensor.reshape(-1, c, h, w)
    image2_tensor = image2_tensor.reshape(-1, c, h, w)

    sum_result = None
    sumsq_result = None
    count_result = None

    with torch.inference_mode():
        for start_idx in tqdm(range(0, image1_tensor.shape[0], latent_batch_size)):
            end_idx = min(start_idx + latent_batch_size, image1_tensor.shape[0])
            batch1 = image1_tensor[start_idx:end_idx].to(device=device, dtype=torch.float32)
            batch2 = image2_tensor[start_idx:end_idx].to(device=device, dtype=torch.float32)

            batch1 = transform(batch1)
            batch2 = transform(batch2)

            latent_features1 = model(batch1)
            latent_features2 = model(batch2)
            diff_sq = (latent_features1 - latent_features2) ** 2
            diff_sq = diff_sq.reshape(diff_sq.shape[0], -1)

            frame_ids = torch.arange(start_idx, end_idx, device=device) % n_frames
            if sum_result is None:
                latent_c = diff_sq.shape[1]
                sum_result = torch.zeros(n_frames, device=device, dtype=torch.float64)
                sumsq_result = torch.zeros(n_frames, device=device, dtype=torch.float64)
                count_result = torch.zeros(n_frames, device=device, dtype=torch.float64)

            diff_sq = diff_sq.to(torch.float64)
            sum_result.scatter_add_(0, frame_ids, diff_sq.sum(dim=1))
            sumsq_result.scatter_add_(0, frame_ids, (diff_sq ** 2).sum(dim=1))
            count_result.scatter_add_(
                0,
                frame_ids,
                torch.full((end_idx - start_idx,), latent_c, device=device, dtype=torch.float64),
            )

    mse_result = (sum_result / count_result).detach().cpu().numpy()
    std_result = torch.sqrt(
        torch.clamp(sumsq_result / count_result - sum_result.square() / count_result.square(), min=0)
    ).detach().cpu().numpy()

    mse = {}
    std = {}

    for clip_timestamp in range(len(mse_result)):
        mse[clip_timestamp] = mse_result[clip_timestamp].tolist()
        std[clip_timestamp] = std_result[clip_timestamp].tolist()

    result = {
        "value": mse,
        "value_mean": float(np.mean(mse_result)),
        "value_std": std,
        "video_setting": torch.Size([batch_size * n_frames, c, 299, 299]),
        "video_setting_name": "time, channel, heigth, width",
    }
    return result
