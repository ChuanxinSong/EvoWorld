from typing import Callable, Dict, List, Optional, Union

import PIL.Image
import einops
import torch
from diffusers.utils.torch_utils import randn_tensor

from evoworld.pipeline.pipeline_evoworld import (
    StableVideoDiffusionPipeline,
    StableVideoDiffusionPipelineOutput,
    _append_dims,
    retrieve_timesteps,
)


class StableVideoDiffusionOnlyPluckerPipeline(StableVideoDiffusionPipeline):
    """SVD pipeline variant for first-frame + Plucker conditioning only.

    This keeps the public call shape close to the existing EvoWorld pipeline,
    but intentionally ignores memory/reprojection images so the UNet input is:
    4 noisy latent channels + 4 first-frame latent channels + 6 Plucker channels.
    """

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        sigmas: Optional[List[float]] = None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        encode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        plucker_embedding: Optional[torch.Tensor] = None,
        memorized_plucker_embedding: Optional[torch.Tensor] = None,
        memorized_pixel_values: Optional[torch.Tensor] = None,
        mask_mem: Optional[torch.Tensor] = False,
    ):
        del memorized_plucker_embedding, memorized_pixel_values, mask_mem

        if plucker_embedding is None:
            raise ValueError("`plucker_embedding` is required for the only-plucker pipeline.")

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        encode_chunk_size = encode_chunk_size if encode_chunk_size is not None else decode_chunk_size

        self.check_inputs(image, height, width)

        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
            image = image / 2.0 + 0.5

        device = self._execution_device
        self._guidance_scale = max_guidance_scale

        image_embeddings = self._encode_image(
            image,
            device,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )

        # Stable Video Diffusion was conditioned on fps - 1.
        fps = fps - 1

        image = self.video_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            encode_chunk_size=encode_chunk_size,
        )
        image_latents = image_latents.to(image_embeddings.dtype)
        image_latents = image_latents[:, None].repeat(1, num_frames, 1, 1, 1)

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        plucker_embedding = plucker_embedding.to(device=device, dtype=image_embeddings.dtype)
        plucker_embedding = plucker_embedding.repeat_interleave(num_videos_per_prompt, dim=0)
        if self.do_classifier_free_guidance:
            plucker_embedding = torch.cat([plucker_embedding, plucker_embedding], dim=0)

        conditional_latents = torch.cat([image_latents, plucker_embedding], dim=2)

        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, sigmas
        )

        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            self.unet.config.in_channels,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        self._guidance_scale = guidance_scale

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, conditional_latents], dim=2)

                if latent_model_input.shape[2] != self.unet.config.in_channels:
                    raise ValueError(
                        "Only-plucker UNet channel mismatch: "
                        f"input has {latent_model_input.shape[2]} channels, "
                        f"UNet expects {self.unet.config.in_channels}."
                    )

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        if not output_type == "latent":
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)
