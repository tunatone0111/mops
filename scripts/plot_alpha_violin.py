"""Best AUROC config의 head별 alpha_summary_mean 분포를 violin plot으로 시각화한다."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

font_path = "/home/tunatone0111/.local/share/fonts/Outfit-Regular.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = "Outfit"
plt.rcParams["font.weight"] = "regular"
plt.rcParams["mathtext.fontset"] = "cm"

RESULTS_DIR = Path("results")
METRIC = "alpha_summary_mean"
GROUP_PALETTE = {"Mem": "#f0a59f", "Non-Mem": "#b1c7fb"}

# --- 시각화 대상 config (compute_alpha_metrics.py 결과에서 가져온 best AUROC config) ---
CONFIGS = [
    {
        "name": "SD1.4",
        "memorized_csv": "memorized_v2.csv",
        "unmemorized_csv": "unmemorized_v2.csv",
        "timestep": 0,
        "layer": "mid_attn_0_block_0",
        "title": "SD1.4 — mid_a0, timestep=0, per head",
        "out": "alpha_violin_sd14_mid_a0_ts0.pdf",
    },
    {
        "name": "SD1.4",
        "memorized_csv": "memorized_v2.csv",
        "unmemorized_csv": "unmemorized_v2.csv",
        "timestep": 9,
        "layer": "mid_attn_0_block_0",
        "title": "SD1.4 — mid_a0, timestep=9, per head",
        "out": "alpha_violin_sd14_mid_a0_ts9.pdf",
    },
    {
        "name": "SD1.4",
        "memorized_csv": "memorized_v2.csv",
        "unmemorized_csv": "unmemorized_v2.csv",
        "timestep": 0,
        "layer": "up_3_attn_2_block_0",
        "title": "SD1.4 — u3_a2, timestep=0, per head",
        "out": "alpha_violin_sd14_u3_a2_ts0.pdf",
    },
    {
        "name": "SD1.4",
        "memorized_csv": "memorized_v2.csv",
        "unmemorized_csv": "unmemorized_v2.csv",
        "timestep": 9,
        "layer": "up_3_attn_2_block_0",
        "title": "SD1.4 — u3_a2, timestep=9, per head",
        "out": "alpha_violin_sd14_u3_a2_ts9.pdf",
    },
    {
        "name": "SD2.1",
        "memorized_csv": "memorized_sd21.csv",
        "unmemorized_csv": "unmemorized_sd21.csv",
        "timestep": 0,
        "layer": "down_2_attn_0_block_0",
        "title": "SD2.1 — d2_a0, timestep=0, per head",
        "out": "alpha_violin_sd21_d2_a0_ts0.pdf",
    },
    {
        "name": "SD2.1",
        "memorized_csv": "memorized_sd21.csv",
        "unmemorized_csv": "unmemorized_sd21.csv",
        "timestep": 9,
        "layer": "down_2_attn_0_block_0",
        "title": "SD2.1 — d2_a0, timestep=9, per head",
        "out": "alpha_violin_sd21_d2_a0_ts9.pdf",
    },
    {
        "name": "SD2.1",
        "memorized_csv": "memorized_sd21.csv",
        "unmemorized_csv": "unmemorized_sd21.csv",
        "timestep": 0,
        "layer": "up_1_attn_0_block_0",
        "title": "SD2.1 — u1_a0, timestep=0, per head",
        "out": "alpha_violin_sd21_u1_a0_ts0.pdf",
    },
    {
        "name": "SD2.1",
        "memorized_csv": "memorized_sd21.csv",
        "unmemorized_csv": "unmemorized_sd21.csv",
        "timestep": 9,
        "layer": "up_1_attn_0_block_0",
        "title": "SD2.1 — u1_a0, timestep=9, per head",
        "out": "alpha_violin_sd21_u1_a0_ts9.pdf",
    },
]


def load(memorized_csv: str, unmemorized_csv: str, timestep: int, layer: str) -> pd.DataFrame:
    """지정된 (timestep, layer)의 데이터를 로드한다."""
    cols = ["prompt_idx", "timestep", "layer", "head", METRIC]
    dfs = []
    for fname, group in [(memorized_csv, "Mem"), (unmemorized_csv, "Non-Mem")]:
        df = pd.read_csv(RESULTS_DIR / fname, usecols=cols)
        df["group"] = group
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df[(df["timestep"] == timestep) & (df["layer"] == layer)]


def plot(df: pd.DataFrame, title: str, out_path: Path) -> None:
    """head별 split violin plot을 생성한다."""
    fig, ax = plt.subplots(figsize=(3.2, 1.2), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=1 / 72, h_pad=1 / 72, wspace=0, hspace=0)
    sns.violinplot(
        data=df,
        x="head",
        y=METRIC,
        hue="group",
        split=True,
        inner=None,
        palette=GROUP_PALETTE,
        ax=ax,
        density_norm="width",
        cut=0,
        alpha=0.7,
        linewidth=0.75,
    )
    # stroke 색깔을 palette 원색(alpha=1.0)으로 설정
    palette_colors = list(GROUP_PALETTE.values())
    for i, coll in enumerate(ax.collections):
        if hasattr(coll, "get_facecolor"):
            color = palette_colors[i % len(palette_colors)]
            coll.set_edgecolor(color)
    ax.set_xlabel("Head Index", fontsize=7)
    ax.set_ylabel(r"$A_{\tilde{sp}}^{(h)}$", fontsize=7)
    ax.tick_params(labelsize=5)
    legend = ax.legend(fontsize=5, loc="upper right")
    for i, patch in enumerate(legend.legend_handles):
        if hasattr(patch, "set_edgecolor"):
            patch.set_edgecolor(palette_colors[i % len(palette_colors)])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"저장: {out_path}")


def main() -> None:
    for cfg in CONFIGS:
        df = load(cfg["memorized_csv"], cfg["unmemorized_csv"], cfg["timestep"], cfg["layer"])
        plot(df, cfg["title"], RESULTS_DIR / cfg["out"])


if __name__ == "__main__":
    main()
