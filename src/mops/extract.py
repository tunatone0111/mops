"""Cross-attention 통계 추출 메인 엔트리포인트."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor
from omegaconf import DictConfig

from mops.processor import StatsAttnProcessor
from mops.stats_store import CrossAttentionStatsStore
from mops.token_utils import classify_tokens

CROSS_ATTN_KEY = "attn2"


def _processor_key_to_layer_name(key: str) -> str:
    """
    diffusers의 attn_processor key를 사람이 읽기 좋은 이름으로 변환.

    예시:
    - "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor" → "down_2_attn_1_block_0"
    - "mid_block.attentions.0.transformer_blocks.0.attn2.processor" → "mid_attn_0_block_0"
    - "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor" → "up_1_attn_0_block_0"
    """
    parts = key.split(".")

    if parts[0] in ("down_blocks", "up_blocks"):
        prefix = "down" if parts[0] == "down_blocks" else "up"
        block_index = parts[1]
        attention_index = parts[3]
        transformer_block_index = parts[5]
        return f"{prefix}_{block_index}_attn_{attention_index}_block_{transformer_block_index}"
    elif parts[0] == "mid_block":
        attention_index = parts[2]
        transformer_block_index = parts[4]
        return f"mid_attn_{attention_index}_block_{transformer_block_index}"
    else:
        # 알 수 없는 패턴은 원본 키에서 ".processor" 제거 후 반환
        return key.replace(".processor", "").replace(".", "_")


def install_stats_processors(unet, stats_store: CrossAttentionStatsStore) -> None:
    """UNet의 cross-attention(attn2)에만 StatsAttnProcessor를 설치한다."""
    processors = {}
    for key in unet.attn_processors:
        if CROSS_ATTN_KEY in key:
            layer_name = _processor_key_to_layer_name(key)
            processors[key] = StatsAttnProcessor(stats_store=stats_store, layer_name=layer_name)
        else:
            # self-attention 등은 표준 processor 사용
            processors[key] = AttnProcessor()
    unet.set_attn_processor(processors)


@hydra.main(config_path="../../conf", config_name="extract", version_base=None)
def main(cfg: DictConfig) -> None:
    # 1. 입력 JSON 로드
    json_path = Path(cfg.input.json_path)
    if not json_path.is_absolute():
        json_path = Path(hydra.utils.get_original_cwd()) / json_path
    with open(json_path) as f:
        prompt_items = json.load(f)

    # 2. 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.float16 if cfg.model.dtype == "float16" else torch.float32
    scheduler = DDIMScheduler.from_pretrained(cfg.model.model_id, subfolder="scheduler")
    pipeline = StableDiffusionPipeline.from_pretrained(
        cfg.model.model_id,
        scheduler=scheduler,
        torch_dtype=model_dtype,
        safety_checker=None,
    ).to(device)

    # 3. 통계 저장소 생성 및 processor 설치
    stats_store = CrossAttentionStatsStore()
    install_stats_processors(pipeline.unet, stats_store)

    # 4. 출력 경로 설정
    output_path = Path(cfg.output.csv_path)
    if not output_path.is_absolute():
        output_path = Path(hydra.utils.get_original_cwd()) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 이전 실행 결과가 있으면 덮어쓰기 위해 삭제
    if output_path.exists():
        output_path.unlink()

    # 5. 프롬프트별 생성 루프
    seed = cfg.inference.seed

    def on_step_end(_pipe_instance, step_index, _timestep, callback_kwargs):
        # 다음 스텝을 위해 timestep index 갱신
        stats_store.current_timestep_index = step_index + 1
        return callback_kwargs

    for prompt_index, item in enumerate(prompt_items):
        prompt_text = item["prompt"]
        print(f"[{prompt_index + 1}/{len(prompt_items)}] {prompt_text!r}")

        # 토큰 분류
        token_groups = classify_tokens(pipeline.tokenizer, prompt_text)
        stats_store.current_prompt_index = prompt_index
        stats_store.prompt_token_indices = token_groups["prompt"]
        stats_store.summary_token_indices = token_groups["summary"]

        # 첫 스텝 전 timestep 설정
        stats_store.current_timestep_index = 0

        # 생성 실행
        generator = torch.Generator(device=device).manual_seed(seed)
        pipeline(
            prompt_text,
            num_inference_steps=cfg.inference.num_inference_steps,
            guidance_scale=cfg.inference.guidance_scale,
            generator=generator,
            output_type="latent",
            callback_on_step_end=on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        # 프롬프트 완료 후 즉시 CSV에 append
        n_flushed = stats_store.flush_to_csv(output_path)
        print(f"  → {n_flushed} rows 기록 (누적 파일: {output_path})")

    print(f"\n완료: {output_path}")


if __name__ == "__main__":
    main()
