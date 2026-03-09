"""일반/마스킹 이미지 생성 모듈."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from mops.token_utils import classify_tokens

logger = logging.getLogger(__name__)


def _make_generator(device: torch.device | str, seed: int) -> torch.Generator:
    """재현 가능한 Generator 생성."""
    return torch.Generator(device=device).manual_seed(seed)


def _tokenize(tokenizer, text: str, device: torch.device | str):
    """CLIP 토크나이저로 텍스트를 토크나이즈."""
    return tokenizer(
        text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)


@torch.no_grad()
def _get_uncond_embeds(pipeline: StableDiffusionPipeline) -> torch.Tensor:
    """unconditional embedding 계산 (빈 문자열)."""
    inputs = _tokenize(pipeline.tokenizer, "", pipeline.device)
    return pipeline.text_encoder(inputs.input_ids).last_hidden_state


@torch.no_grad()
def _encode_prompt(pipeline: StableDiffusionPipeline, prompt: str) -> torch.Tensor:
    """프롬프트를 텍스트 인코더로 인코딩하여 임베딩 반환."""
    inputs = _tokenize(pipeline.tokenizer, prompt, pipeline.device)
    return pipeline.text_encoder(inputs.input_ids).last_hidden_state


@torch.no_grad()
def _compute_masked_embeds(pipeline: StableDiffusionPipeline, prompt: str) -> torch.Tensor:
    """프롬프트 토큰을 제로마스킹한 임베딩 반환 (BOS/EOS/PAD는 유지)."""
    embeds = _encode_prompt(pipeline, prompt).clone()
    token_groups = classify_tokens(pipeline.tokenizer, prompt)
    embeds[:, token_groups["prompt"], :] = 0.0
    return embeds


def generate_normal_image(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
) -> Image.Image:
    """프롬프트로 일반 이미지 생성."""
    generator = _make_generator(pipeline.device, seed)
    result = pipeline(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return result.images[0]


@torch.no_grad()
def generate_mitigation_image(
    pipeline: StableDiffusionPipeline,
    masked_embeds: torch.Tensor,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt_embeds: torch.Tensor,
) -> Image.Image:
    """사전 계산된 마스킹 임베딩으로 이미지 생성."""
    generator = _make_generator(pipeline.device, seed)
    result = pipeline(
        prompt=None,
        prompt_embeds=masked_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return result.images[0]


@torch.no_grad()
def generate_switching_image(
    pipeline: StableDiffusionPipeline,
    original_embeds: torch.Tensor,
    masked_embeds: torch.Tensor,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    switch_ratio: float,
    negative_prompt_embeds: torch.Tensor,
) -> Image.Image:
    """커스텀 디노이징 루프로 switching 이미지 생성.

    전반부(switch_step 이전)는 마스킹된 패러프레이즈 임베딩,
    후반부(switch_step 이후)는 원본 프롬프트 풀 임베딩을 사용.
    """
    device = pipeline.device
    dtype = pipeline.unet.dtype

    # 스케줄러 설정
    scheduler = pipeline.scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    switch_step = int(num_inference_steps * switch_ratio)

    # 초기 latent noise 생성
    generator = _make_generator(device, seed)
    unet_cfg = pipeline.unet.config
    latent_shape = (1, unet_cfg.in_channels, unet_cfg.sample_size, unet_cfg.sample_size)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    # 디노이징 루프
    for step_idx, t in enumerate(timesteps):
        cond_embeds = masked_embeds if step_idx < switch_step else original_embeds

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        prompt_embeds_combined = torch.cat([negative_prompt_embeds, cond_embeds])

        noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds_combined).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # VAE 디코딩
    latents = 1 / pipeline.vae.config.scaling_factor * latents
    image = pipeline.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return pipeline.numpy_to_pil(image)[0]


def generate_all_images(
    pipeline: StableDiffusionPipeline,
    prompt_groups: list[dict],
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    switch_ratio: float,
    output_dir: Path,
) -> dict[int, dict]:
    """
    모든 프롬프트 그룹에 대해 4가지 타입의 이미지 생성.

    Returns:
        {prompt_idx: {"original": Path, "paraphrases": [Path, ...],
                      "mitigations": [Path, ...], "switchings": [Path, ...]}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: dict[int, dict] = {}

    # unconditional embedding은 한 번만 계산
    uncond_embeds = _get_uncond_embeds(pipeline)

    for prompt_idx, group in enumerate(prompt_groups):
        original = group["original"]
        paraphrases = group["paraphrases"]
        prompt_dir = output_dir / f"prompt_{prompt_idx:03d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        paths: dict = {"original": None, "paraphrases": [], "mitigations": [], "switchings": []}

        # (A) 원본 이미지
        original_path = prompt_dir / "original.png"
        if not original_path.exists():
            logger.info(f"[{prompt_idx}] 원본 이미지 생성: {original!r}")
            img = generate_normal_image(pipeline, original, seed, num_inference_steps, guidance_scale)
            img.save(original_path)
        paths["original"] = original_path

        # 원본 프롬프트 임베딩은 그룹당 한 번만 계산
        original_embeds = _encode_prompt(pipeline, original)

        for para_idx, para_prompt in enumerate(paraphrases):
            # (B) Paraphrase 이미지
            para_path = prompt_dir / f"paraphrase_{para_idx:02d}.png"
            if not para_path.exists():
                logger.info(f"[{prompt_idx}] 패러프레이즈 이미지 생성 ({para_idx}): {para_prompt!r}")
                img = generate_normal_image(pipeline, para_prompt, seed, num_inference_steps, guidance_scale)
                img.save(para_path)
            paths["paraphrases"].append(para_path)

            # 마스킹 임베딩은 paraphrase당 한 번만 계산하여 mitigation/switching에서 재사용
            masked_embeds = _compute_masked_embeds(pipeline, para_prompt)

            # (C) Mitigation 이미지
            mit_path = prompt_dir / f"mitigation_{para_idx:02d}.png"
            if not mit_path.exists():
                logger.info(f"[{prompt_idx}] 미티게이션 이미지 생성 ({para_idx}): {para_prompt!r}")
                img = generate_mitigation_image(
                    pipeline, masked_embeds, seed, num_inference_steps, guidance_scale, uncond_embeds
                )
                img.save(mit_path)
            paths["mitigations"].append(mit_path)

            # (D) Switching 이미지
            sw_path = prompt_dir / f"switching_{para_idx:02d}.png"
            if not sw_path.exists():
                logger.info(f"[{prompt_idx}] 스위칭 이미지 생성 ({para_idx}): {para_prompt!r}")
                img = generate_switching_image(
                    pipeline,
                    original_embeds,
                    masked_embeds,
                    seed,
                    num_inference_steps,
                    guidance_scale,
                    switch_ratio,
                    uncond_embeds,
                )
                img.save(sw_path)
            paths["switchings"].append(sw_path)

        image_paths[prompt_idx] = paths
        logger.info(f"[{prompt_idx + 1}/{len(prompt_groups)}] 이미지 생성 완료")

    return image_paths
