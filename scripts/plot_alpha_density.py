"""특정 (timestep, layer, head)에서 alpha_summary_mean의 density plot 생성."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path("results")
METRIC = "alpha_summary_mean"
GROUP_PALETTE = {"memorized": "salmon", "unmemorized": "cornflowerblue"}

# --- 시각화 대상 config ---
CONFIGS = [
    {
        "memorized_csv": "memorized_sd21.csv",
        "unmemorized_csv": "unmemorized_sd21.csv",
        "timestep": 0,
        "layer": "down_2_attn_0_block_0",
        "head": 3,
        "title": "SD2.1 — d2_a0, head=3, timestep=0",
        "out": "alpha_density_sd21_d2a0_h3_ts0.png",
    },
]

# --- 수정 가능한 plot 설정 ---
PLOT_CFG = {
    "figsize": (8, 5),
    "xlabel": r"$\alpha_{\mathrm{summary}}$",
    "ylabel": "Density",
    "alpha": 0.4,
    "dpi": 150,
}


def load(cfg: dict) -> pd.DataFrame:
    """지정된 (timestep, layer, head)의 데이터를 로드한다."""
    cols = ["prompt_idx", "timestep", "layer", "head", METRIC]
    dfs = []
    for fname, group in [(cfg["memorized_csv"], "memorized"), (cfg["unmemorized_csv"], "unmemorized")]:
        df = pd.read_csv(RESULTS_DIR / fname, usecols=cols)
        df["group"] = group
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df[(df["timestep"] == cfg["timestep"]) & (df["layer"] == cfg["layer"]) & (df["head"] == cfg["head"])]


def plot(df: pd.DataFrame, cfg: dict) -> None:
    """density plot을 생성한다."""
    fig, ax = plt.subplots(figsize=PLOT_CFG["figsize"], constrained_layout=True)

    for group, color in GROUP_PALETTE.items():
        values = df[df["group"] == group][METRIC]
        sns.kdeplot(values, ax=ax, color=color, label=group, fill=True, alpha=PLOT_CFG["alpha"], linewidth=1.5)

    ax.set_xlabel(PLOT_CFG["xlabel"])
    ax.set_ylabel(PLOT_CFG["ylabel"])
    ax.set_title(cfg["title"])
    ax.legend()

    out = RESULTS_DIR / cfg["out"]
    fig.savefig(out, dpi=PLOT_CFG["dpi"])
    plt.close(fig)
    print(f"저장: {out}")


def main() -> None:
    for cfg in CONFIGS:
        df = load(cfg)
        plot(df, cfg)


if __name__ == "__main__":
    main()
