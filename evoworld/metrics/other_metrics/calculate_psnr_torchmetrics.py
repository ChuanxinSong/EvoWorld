import numpy as np
import torch
from tqdm import tqdm


def _load_torchmetrics_psnr():
    try:
        from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
    except ImportError as exc:
        raise ImportError(
            "torchmetrics is required for calculate_psnr_torchmetrics. "
            "Install/fix torchmetrics and its dependencies, or switch back to "
            "evoworld.metrics.other_metrics.calculate_psnr."
        ) from exc

    return peak_signal_noise_ratio


def calculate_psnr(videos1, videos2, device=None, batch_size=64):
    print("calculate_psnr_torchmetrics...")

    # videos [batch_size, timestamps, channel, h, w], value range [0, 1]
    assert videos1.shape == videos2.shape

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    peak_signal_noise_ratio = _load_torchmetrics_psnr()

    num_videos, num_frames = videos1.shape[:2]
    flat_videos1 = videos1.reshape(num_videos * num_frames, *videos1.shape[2:])
    flat_videos2 = videos2.reshape(num_videos * num_frames, *videos2.shape[2:])

    psnr_scores = []
    for start_idx in tqdm(range(0, flat_videos1.shape[0], batch_size)):
        end_idx = min(start_idx + batch_size, flat_videos1.shape[0])
        batch1 = flat_videos1[start_idx:end_idx].to(device)
        batch2 = flat_videos2[start_idx:end_idx].to(device)

        with torch.no_grad():
            batch_scores = peak_signal_noise_ratio(
                batch1,
                batch2,
                data_range=1.0,
                reduction="none",
                dim=(1, 2, 3),
            )
            batch_scores = torch.where(
                torch.isinf(batch_scores),
                torch.full_like(batch_scores, 100),
                batch_scores,
            )
        psnr_scores.append(batch_scores.detach().cpu())

    psnr_results = torch.cat(psnr_scores).reshape(num_videos, num_frames).numpy()

    psnr = {}
    psnr_std = {}

    for clip_timestamp in range(num_frames):
        psnr[clip_timestamp] = np.mean(psnr_results[:, clip_timestamp])
        psnr_std[clip_timestamp] = np.std(psnr_results[:, clip_timestamp])

    result = {
        "value": psnr,
        "value_mean": float(np.mean(psnr_results)),
        "value_std": psnr_std,
        "video_setting": videos1[0].shape,
        "video_setting_name": "time, channel, heigth, width",
    }

    return result


def main():
    number_of_videos = 8
    video_length = 50
    channel = 3
    size = 64
    videos1 = torch.zeros(number_of_videos, video_length, channel, size, size)
    videos2 = torch.zeros(number_of_videos, video_length, channel, size, size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import json

    result = calculate_psnr(videos1, videos2, device=device)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
