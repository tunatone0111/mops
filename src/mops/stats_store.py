"""Cross-attention 통계 저장소."""

from __future__ import annotations


class CrossAttentionStatsStore:
    """
    프로세서가 통계를 기록하는 저장소.

    csv_records의 각 원소가 CSV 한 행에 대응한다.
    """

    def __init__(self) -> None:
        self.csv_records: list[dict] = []
        self.current_timestep_index: int = 0
        self.current_prompt_index: int = 0
        self.prompt_token_indices: list[int] = []
        self.summary_token_indices: list[int] = []

    def add_record(self, layer_name: str, head_index: int, **stats: float) -> None:
        self.csv_records.append(
            {
                "prompt_idx": self.current_prompt_index,
                "timestep": self.current_timestep_index,
                "n_prompt_tokens": len(self.prompt_token_indices),
                "layer": layer_name,
                "head": head_index,
                **stats,
            }
        )
