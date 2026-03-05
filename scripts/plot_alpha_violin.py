"""Best AUROC config의 head별 alpha_summary_mean 분포를 violin plot으로 시각화한다."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path("results")
METRIC = "alpha_summary_mean"
GROUP_PALETTE = {"memorized": "salmon", "unmemorized": "cornflowerblue"}

# --- 시각화 대상 config (compute_alpha_metrics.py 결과에서 가져온 best AUROC config) ---
CONFIGS = [
    {
        "name": "SD1.4",
        "memorized_csv": "memorized_v2.csv",
        "unmemorized_csv": "unmemorized_v2.csv",
        "timestep": 0,
        "layer": "mid_attn_0_block_0",
        "title": "SD1.4 — mid_a0, timestep=0, per head",
        "out": "alpha_violin_sd14_mid_a0_ts0.png",
    },
    {
        "name": "SD1.4",
        "memorized_csv": "memorized_v2.csv",
        "unmemorized_csv": "unmemorized_v2.csv",
        "timestep": 9,
        "layer": "mid_attn_0_block_0",
        "title": "SD1.4 — mid_a0, timestep=9, per head",
        "out": "alpha_violin_sd14_mid_a0_ts9.png",
    },
    {
        "name": "SD1.4",
        "memorized_csv": "memorized_v2.csv",
        "unmemorized_csv": "unmemorized_v2.csv",
        "timestep": 0,
        "layer": "up_3_attn_2_block_0",
        "title": "SD1.4 — u3_a2, timestep=0, per head",
        "out": "alpha_violin_sd14_u3_a2_ts0.png",
    },
    {
        "name": "SD1.4",
        "memorized_csv": "memorized_v2.csv",
        "unmemorized_csv": "unmemorized_v2.csv",
        "timestep": 9,
        "layer": "up_3_attn_2_block_0",
        "title": "SD1.4 — u3_a2, timestep=9, per head",
        "out": "alpha_violin_sd14_u3_a2_ts9.png",
    },
    {
        "name": "SD2.1",
        "memorized_csv": "memorized_sd21.csv",
        "unmemorized_csv": "unmemorized_sd21.csv",
        "timestep": 0,
        "layer": "down_2_attn_0_block_0",
        "title": "SD2.1 — d2_a0, timestep=0, per head",
        "out": "alpha_violin_sd21_d2_a0_ts0.png",
    },
    {
        "name": "SD2.1",
        "memorized_csv": "memorized_sd21.csv",
        "unmemorized_csv": "unmemorized_sd21.csv",
        "timestep": 9,
        "layer": "down_2_attn_0_block_0",
        "title": "SD2.1 — d2_a0, timestep=9, per head",
        "out": "alpha_violin_sd21_d2_a0_ts9.png",
    },
    {
        "name": "SD2.1",
        "memorized_csv": "memorized_sd21.csv",
        "unmemorized_csv": "unmemorized_sd21.csv",
        "timestep": 0,
        "layer": "up_1_attn_0_block_0",
        "title": "SD2.1 — u1_a0, timestep=0, per head",
        "out": "alpha_violin_sd21_u1_a0_ts0.png",
    },
    {
        "name": "SD2.1",
        "memorized_csv": "memorized_sd21.csv",
        "unmemorized_csv": "unmemorized_sd21.csv",
        "timestep": 9,
        "layer": "up_1_attn_0_block_0",
        "title": "SD2.1 — u1_a0, timestep=9, per head",
        "out": "alpha_violin_sd21_u1_a0_ts9.png",
    },
]


def load(memorized_csv: str, unmemorized_csv: str, timestep: int, layer: str) -> pd.DataFrame:
    """지정된 (timestep, layer)의 데이터를 로드한다."""
    cols = ["prompt_idx", "timestep", "layer", "head", METRIC]
    dfs = []
    for fname, group in [(memorized_csv, "memorized"), (unmemorized_csv, "unmemorized")]:
        df = pd.read_csv(RESULTS_DIR / fname, usecols=cols)
        df["group"] = group
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df[(df["timestep"] == timestep) & (df["layer"] == layer)]


def plot(df: pd.DataFrame, title: str, out_path: Path) -> None:
    """head별 split violin plot을 생성한다."""
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sns.violinplot(
        data=df,
        x="head",
        y=METRIC,
        hue="group",
        split=True,
        inner="quart",
        palette=GROUP_PALETTE,
        ax=ax,
        density_norm="width",
        cut=0,
    )
    ax.set_title(title)
    ax.set_xlabel("Head")
    ax.set_ylabel(r"$\alpha_{\mathrm{summary}}$")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"저장: {out_path}")


def main() -> None:
    for cfg in CONFIGS:
        df = load(cfg["memorized_csv"], cfg["unmemorized_csv"], cfg["timestep"], cfg["layer"])
        plot(df, cfg["title"], RESULTS_DIR / cfg["out"])


if __name__ == "__main__":
    main()
