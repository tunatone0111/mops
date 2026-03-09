"""CLIP EOS 토큰 cosine similarity 계산."""

from __future__ import annotations

import logging

import pandas as pd
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _find_eos_positions(input_ids: torch.Tensor, eos_token_id: int) -> list[int]:
    """배치 내 각 시퀀스에서 EOS 토큰 위치를 반환."""
    positions = []
    for i in range(input_ids.shape[0]):
        seq = input_ids[i].tolist()
        # BOS(0번) 이후 처음 나타나는 EOS 위치
        pos = seq.index(eos_token_id) if eos_token_id in seq else len(seq) - 1
        positions.append(pos)
    return positions


@torch.no_grad()
def compute_eos_similarities(
    text_encoder,
    tokenizer,
    prompt_groups: list[dict],
    device: torch.device | str,
) -> pd.DataFrame:
    """
    각 프롬프트 그룹에서 원본과 패러프레이즈의 EOS 토큰 cosine similarity 계산.

    Args:
        text_encoder: CLIP text encoder (pipeline.text_encoder)
        tokenizer: CLIP tokenizer (pipeline.tokenizer)
        prompt_groups: [{"original": "...", "paraphrases": ["...", ...]}, ...]
        device: 연산 디바이스

    Returns:
        DataFrame with columns: [prompt_idx, original_prompt, paraphrase_idx, paraphrase_prompt, eos_cosine_similarity]
    """
    eos_token_id = tokenizer.eos_token_id
    rows = []

    for prompt_idx, group in enumerate(prompt_groups):
        original = group["original"]
        paraphrases = group["paraphrases"]
        all_prompts = [original, *paraphrases]

        # 한번에 토크나이즈 + 인코딩
        inputs = tokenizer(
            all_prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        hidden_states = text_encoder(inputs.input_ids).last_hidden_state.float()

        # input_ids에서 직접 EOS 위치 찾기 (이중 토크나이즈 방지)
        eos_positions = _find_eos_positions(inputs.input_ids, eos_token_id)
        eos_embeddings = [hidden_states[i, eos_positions[i]] for i in range(len(all_prompts))]

        original_eos = eos_embeddings[0].unsqueeze(0)

        for para_idx, para_prompt in enumerate(paraphrases):
            para_eos = eos_embeddings[para_idx + 1].unsqueeze(0)
            cos_sim = F.cosine_similarity(original_eos, para_eos, dim=-1).item()

            rows.append(
                {
                    "prompt_idx": prompt_idx,
                    "original_prompt": original,
                    "paraphrase_idx": para_idx,
                    "paraphrase_prompt": para_prompt,
                    "eos_cosine_similarity": cos_sim,
                }
            )

        logger.info(f"[{prompt_idx + 1}/{len(prompt_groups)}] EOS 유사도 계산 완료: {original!r}")

    return pd.DataFrame(rows)
