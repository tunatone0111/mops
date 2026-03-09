"""OpenAI API를 이용한 프롬프트 패러프레이즈 생성."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5

# "1. xxx", "2. xxx" 형태의 번호 리스트 파싱
_NUMBERED_LINE_RE = re.compile(r"^\d+\.\s*(.+)$", re.MULTILINE)


def _parse_response(content: str, n: int) -> list[str]:
    """API 응답을 파싱하여 패러프레이즈 리스트로 변환."""
    # JSON 배열/딕셔너리 시도
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [str(x) for x in parsed[:n]]
        for value in parsed.values():
            if isinstance(value, list):
                return [str(x) for x in value[:n]]
    except (json.JSONDecodeError, AttributeError):
        pass

    # 번호 리스트 형태 ("1. xxx\n2. yyy") 파싱
    matches = _NUMBERED_LINE_RE.findall(content)
    if matches:
        return [m.strip() for m in matches[:n]]

    raise ValueError(f"예상하지 못한 응답 형식: {content}")


def generate_paraphrases(
    client: OpenAI,
    prompt: str,
    n: int,
    model: str,
    system_prompt: str,
    user_prompt_template: str,
) -> list[str]:
    """단일 프롬프트에 대해 n개의 패러프레이즈를 생성 (재시도 포함)."""
    user_message = user_prompt_template.format(n=n, prompt=prompt)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.8,
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("API 응답 content가 None")

            return _parse_response(content, n)

        except Exception:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"API 호출 실패 (시도 {attempt + 1}/{MAX_RETRIES}), {RETRY_DELAY}초 후 재시도")
                time.sleep(RETRY_DELAY)
            else:
                raise


def paraphrase_all(
    json_path: Path,
    n: int,
    model: str,
    system_prompt: str,
    user_prompt_template: str,
    cache_path: Path,
) -> list[dict]:
    """
    JSON 파일의 모든 프롬프트에 대해 패러프레이즈를 생성하고 캐싱.
    중단 후 재실행 시 부분 캐시에서 이어서 진행.

    Returns:
        [{"original": "...", "paraphrases": ["...", ...]}, ...]
    """
    # 입력 프롬프트 로드
    with open(json_path) as f:
        prompt_items = json.load(f)

    # 캐시가 있으면 로드 및 완료 여부 확인
    results: list[dict] = []
    if cache_path.exists():
        with open(cache_path) as f:
            results = json.load(f)
        if len(results) >= len(prompt_items):
            logger.info(f"캐시에서 {len(results)}개 프롬프트 그룹 로드 (완료): {cache_path}")
            return results
        logger.info(f"부분 캐시 발견: {len(results)}/{len(prompt_items)}개 완료, 이어서 진행")

    client = OpenAI()  # OPENAI_API_KEY 환경변수 사용
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    for i in range(len(results), len(prompt_items)):
        original = prompt_items[i]["prompt"]
        logger.info(f"[{i + 1}/{len(prompt_items)}] 패러프레이즈 생성: {original!r}")

        paraphrases = generate_paraphrases(client, original, n, model, system_prompt, user_prompt_template)
        results.append({"original": original, "paraphrases": paraphrases})

        # 진행 중 캐시 저장 (중단 방지)
        with open(cache_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"패러프레이즈 생성 완료: {len(results)}개 프롬프트, 캐시 저장: {cache_path}")
    return results
