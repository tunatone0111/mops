"""Cross-attention 통계 저장소."""

from __future__ import annotations

import csv
from pathlib import Path


class CrossAttentionStatsStore:
    """
    프로세서가 통계를 기록하는 저장소.

    csv_records에 쌓인 레코드를 flush_to_csv()로 파일에 append 할 수 있다.
    """

    def __init__(self) -> None:
        self.csv_records: list[dict] = []
        self.current_timestep_index: int = 0
        self.current_prompt_index: int = 0
        self.prompt_token_indices: list[int] = []
        self.summary_token_indices: list[int] = []
        self._csv_header_written: bool = False

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

    def flush_to_csv(self, path: Path) -> int:
        """쌓인 레코드를 CSV 파일에 append하고 버퍼를 비운다. 기록한 행 수를 반환."""
        if not self.csv_records:
            return 0

        n_rows = len(self.csv_records)
        fieldnames = list(self.csv_records[0].keys())
        write_header = not self._csv_header_written

        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerows(self.csv_records)

        self.csv_records.clear()
        return n_rows
