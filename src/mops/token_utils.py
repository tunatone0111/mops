"""CLIP 토크나이저 기반 토큰 인덱스 분류 유틸리티."""

from __future__ import annotations


def classify_tokens(tokenizer, prompt: str) -> dict[str, list[int]]:
    """
    CLIP 토크나이저 기준 토큰 인덱스를 3그룹으로 분류.

    - beginning: <|startoftext|> (BOS)
    - prompt: 실제 단어 토큰
    - summary: <|endoftext|> + padding
    """
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    token_ids = encoded.input_ids[0].tolist()

    groups: dict[str, list[int]] = {"beginning": [], "prompt": [], "summary": []}
    found_eos = False
    for index, token_id in enumerate(token_ids):
        if token_id == bos_id:
            groups["beginning"].append(index)
        elif token_id == eos_id or found_eos:
            groups["summary"].append(index)
            found_eos = True
        else:
            groups["prompt"].append(index)
    return groups
