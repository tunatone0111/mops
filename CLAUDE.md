# MOPS — Cross Attention Map 추출·분석

## Project Goal

Stable Diffusion(HuggingFace diffusers 기반) UNet 내부의 cross attention map을 추출·저장·분석한다.

## Tech Stack

- Python 3.11
- 패키지 관리: uv
- 설정 관리: hydra-core (OmegaConf)
- 모델: HuggingFace diffusers, PyTorch

## Code Convention

- 문서/주석: 한국어
- 코드(변수명, 함수명 등): 영어
- Formatter: `ruff format` (line-length=120)
- Linter: `ruff check`
- 코드 작성 완료 후 `/simplify` 스킬 사용

## Git Convention

- Conventional Commits 스타일 커밋 메시지 (e.g. `feat:`, `fix:`, `docs:`, `refactor:`)

## Test

- pytest 사용

## Rules

- When using python, always use uv for package management and running scripts.