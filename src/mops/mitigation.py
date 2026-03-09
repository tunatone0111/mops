"""Paraphrase 기반 memorization mitigation 파이프라인 메인 엔트리포인트."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from dotenv import load_dotenv
from omegaconf import DictConfig

from mops.eos_similarity import compute_eos_similarities
from mops.masked_generation import generate_all_images
from mops.metrics import compute_all_metrics
from mops.paraphrase import paraphrase_all

logger = logging.getLogger(__name__)


def _resolve_path(path_str: str) -> Path:
    """상대 경로를 Hydra 원본 작업 디렉토리 기준으로 해석."""
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(hydra.utils.get_original_cwd()) / path
    return path


@hydra.main(config_path="../../conf", config_name="mitigation", version_base=None)
def main(cfg: DictConfig) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 경로 해석
    json_path = _resolve_path(cfg.input.json_path)
    cache_path = _resolve_path(cfg.paraphrase.cache_path)
    output_dir = _resolve_path(cfg.output.dir)
    images_dir = _resolve_path(cfg.output.images_dir)
    metrics_csv = _resolve_path(cfg.output.metrics_csv)
    eos_csv = _resolve_path(cfg.output.eos_similarity_csv)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Paraphrase 생성 (GPU 불필요) ──
    logger.info("=== Step 1: Paraphrase 생성 ===")
    prompt_groups = paraphrase_all(
        json_path=json_path,
        n=cfg.paraphrase.num_paraphrases,
        model=cfg.paraphrase.openai_model,
        system_prompt=cfg.paraphrase.system_prompt,
        user_prompt_template=cfg.paraphrase.user_prompt_template,
        cache_path=cache_path,
    )

    # ── Step 2: SD 파이프라인 로드 ──
    logger.info("=== Step 2: Stable Diffusion 파이프라인 로드 ===")
    model_dtype = torch.float16 if cfg.model.dtype == "float16" else torch.float32
    scheduler = DDIMScheduler.from_pretrained(cfg.model.model_id, subfolder="scheduler")
    pipeline = StableDiffusionPipeline.from_pretrained(
        cfg.model.model_id,
        scheduler=scheduler,
        torch_dtype=model_dtype,
        safety_checker=None,
    ).to(device)

    # ── Step 3: EOS cosine similarity 계산 ──
    logger.info("=== Step 3: EOS cosine similarity 계산 ===")
    eos_df = compute_eos_similarities(
        text_encoder=pipeline.text_encoder,
        tokenizer=pipeline.tokenizer,
        prompt_groups=prompt_groups,
        device=device,
    )
    eos_csv.parent.mkdir(parents=True, exist_ok=True)
    eos_df.to_csv(eos_csv, index=False)
    logger.info(f"EOS 유사도 저장: {eos_csv}")

    # ── Step 4: 이미지 생성 (원본 + paraphrase + mitigation) ──
    logger.info("=== Step 4: 이미지 생성 ===")
    image_paths = generate_all_images(
        pipeline=pipeline,
        prompt_groups=prompt_groups,
        seed=cfg.inference.seed,
        num_inference_steps=cfg.inference.num_inference_steps,
        guidance_scale=cfg.inference.guidance_scale,
        switch_ratio=cfg.inference.switch_ratio,
        output_dir=images_dir,
    )

    # ── Step 5: 메트릭 계산 (GPU 메모리 절약: pipeline 해제 후) ──
    logger.info("=== Step 5: 메트릭 계산 (SSCD + CLIP Score) ===")
    del pipeline
    torch.cuda.empty_cache()

    metrics_df = compute_all_metrics(
        image_paths=image_paths,
        prompt_groups=prompt_groups,
        device=device,
    )
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f"메트릭 저장: {metrics_csv}")

    logger.info("=== 파이프라인 완료 ===")


if __name__ == "__main__":
    main()
