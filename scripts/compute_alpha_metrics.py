"""alpha_summary_mean 기반 memorized vs unmemorized 분류 성능 지표 산출.

Layer-level (head 평균) 및 per-head 단위로 AUROC, ACC, TPR@1%FPR, TPR@3%FPR을 계산한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

RESULTS_DIR = Path("results")
TIMESTEPS = [0, 9]
METRIC = "alpha_summary_mean"
METRICS = ["AUROC", "ACC", "TPR@1%FPR", "TPR@3%FPR"]
LOAD_COLUMNS = ["prompt_idx", "timestep", "layer", "head", METRIC]

LAYER_ORDER = [
    "down_0_attn_0_block_0",
    "down_0_attn_1_block_0",
    "down_1_attn_0_block_0",
    "down_1_attn_1_block_0",
    "down_2_attn_0_block_0",
    "down_2_attn_1_block_0",
    "mid_attn_0_block_0",
    "up_1_attn_0_block_0",
    "up_1_attn_1_block_0",
    "up_1_attn_2_block_0",
    "up_2_attn_0_block_0",
    "up_2_attn_1_block_0",
    "up_2_attn_2_block_0",
    "up_3_attn_0_block_0",
    "up_3_attn_1_block_0",
    "up_3_attn_2_block_0",
]


def _shorten(name: str) -> str:
    """down_0_attn_0_block_0 → d0_a0"""
    import re

    m = re.match(r"(down|mid|up)_?(\d*)_attn_(\d+)_block_\d+", name)
    if not m:
        return name
    prefix, idx, attn_idx = m.group(1), m.group(2), m.group(3)
    if prefix == "mid":
        return f"mid_a{attn_idx}"
    short = {"down": "d", "up": "u"}[prefix]
    return f"{short}{idx}_a{attn_idx}"


def load_raw(memorized_csv: str, unmemorized_csv: str) -> pd.DataFrame:
    """CSV 로드 → timestep 필터링된 raw DataFrame 반환."""
    dfs = []
    for fname, label in [(memorized_csv, 1), (unmemorized_csv, 0)]:
        df = pd.read_csv(RESULTS_DIR / fname, usecols=LOAD_COLUMNS)
        df["label"] = label
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df[df["timestep"].isin(TIMESTEPS)]


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """주어진 FPR 이하에서 최대 TPR을 반환한다."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # target_fpr 이하인 지점 중 최대 TPR
    valid = fpr <= target_fpr
    if not valid.any():
        return 0.0
    return float(tpr[valid][-1])


def optimal_accuracy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """최적 threshold에서의 accuracy를 반환한다."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # Youden's J = TPR - FPR 최대화
    j = tpr - fpr
    best_idx = np.argmax(j)
    best_thresh = thresholds[best_idx]
    preds = (y_score >= best_thresh).astype(int)
    return float(accuracy_score(y_true, preds))


def _compute_single(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """단일 (y_true, y_score)에 대해 AUROC, ACC, TPR@1%FPR, TPR@3%FPR을 계산한다.

    score를 negate하여 alpha가 낮을수록 memorized(positive)로 판별한다.
    """
    neg_score = -y_score
    return {
        "AUROC": roc_auc_score(y_true, neg_score),
        "ACC": optimal_accuracy(y_true, neg_score),
        "TPR@1%FPR": tpr_at_fpr(y_true, neg_score, 0.01),
        "TPR@3%FPR": tpr_at_fpr(y_true, neg_score, 0.03),
    }



def compute_layer_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    """각 (timestep, layer)에서 head 평균 score로 지표를 계산한다."""
    df = raw.groupby(["prompt_idx", "timestep", "layer", "label"], as_index=False)[METRIC].mean()
    rows = []
    for ts in TIMESTEPS:
        for layer in LAYER_ORDER:
            sub = df[(df["timestep"] == ts) & (df["layer"] == layer)]
            if sub.empty:
                continue
            row = {"timestep": ts, "layer": _shorten(layer)}
            row.update(_compute_single(sub["label"].values, sub[METRIC].values))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_head_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    """각 (timestep, layer, head)에서 개별 head score로 지표를 계산한다."""
    heads = sorted(raw["head"].unique())
    rows = []
    for ts in TIMESTEPS:
        for layer in LAYER_ORDER:
            for head in heads:
                sub = raw[(raw["timestep"] == ts) & (raw["layer"] == layer) & (raw["head"] == head)]
                if sub.empty:
                    continue
                row = {"timestep": ts, "layer": _shorten(layer), "head": head}
                row.update(_compute_single(sub["label"].values, sub[METRIC].values))
                rows.append(row)
    return pd.DataFrame(rows)


def _print_best(result: pd.DataFrame, label_cols: list[str]) -> None:
    """AUROC 및 TPR@1%FPR 최대 config의 모든 metric을 출력한다."""
    for target in ["AUROC", "TPR@1%FPR"]:
        best = result.loc[result[target].idxmax()]
        loc = ", ".join(f"{c}={best[c]}" for c in label_cols)
        print(f"\nBest {target} config: timestep={best['timestep']}, {loc}")
        for metric in METRICS:
            print(f"  {metric:12s}: {best[metric]:.4f}")


def run(memorized_csv: str, unmemorized_csv: str, name: str) -> None:
    """한 데이터셋에 대해 layer-level 및 per-head 분류 성능 지표를 계산·출력한다."""
    raw = load_raw(memorized_csv, unmemorized_csv)
    tag = name.lower().replace(" ", "_").replace(".", "")

    # --- Layer-level (head 평균) ---
    layer_result = compute_layer_metrics(raw)
    layer_display = ["layer", *METRICS]

    for ts in TIMESTEPS:
        sub = layer_result[layer_result["timestep"] == ts]
        print(f"\n{'=' * 70}")
        print(f"  {name} — Layer-level (head 평균), timestep={ts}")
        print(f"{'=' * 70}")
        print(sub[layer_display].to_string(index=False, float_format="{:.4f}".format))
        _print_best(sub, label_cols=["layer"])

    out_layer = RESULTS_DIR / f"alpha_summary_classification_{tag}_layer.csv"
    layer_result.to_csv(out_layer, index=False)
    print(f"저장: {out_layer}")

    # --- Per-head ---
    head_result = compute_head_metrics(raw)
    head_display = ["layer", "head", *METRICS]

    for ts in TIMESTEPS:
        sub = head_result[head_result["timestep"] == ts]
        print(f"\n{'=' * 70}")
        print(f"  {name} — Per-head, timestep={ts}")
        print(f"{'=' * 70}")
        print(sub[head_display].to_string(index=False, float_format="{:.4f}".format))
        _print_best(sub, label_cols=["layer", "head"])

    out_head = RESULTS_DIR / f"alpha_summary_classification_{tag}_perhead.csv"
    head_result.to_csv(out_head, index=False)
    print(f"저장: {out_head}")


def main() -> None:
    run("memorized_v2.csv", "unmemorized_v2.csv", "SD1.4")
    run("memorized_sd21.csv", "unmemorized_sd21.csv", "SD2.1")
    run("memorized_rv14.csv", "unmemorized_rv14.csv", "RV1.4")


if __name__ == "__main__":
    main()
