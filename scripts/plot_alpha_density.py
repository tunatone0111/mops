"""특정 (timestep, layer, head)에서 alpha_summary_mean의 density plot 생성."""

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

# --- 시각화 대상 config ---
CONFIGS = [
    {
        "memorized_csv": "memorized_sd21.csv",
        "unmemorized_csv": "unmemorized_sd21.csv",
        "timestep": 0,
        "layer": "down_2_attn_0_block_0",
        "head": 3,
        "title": "SD2.1 — d2_a0, head=3, timestep=0",
        "out": "alpha_density_sd21_d2a0_h3_ts0.pdf",
    },
]

# --- 수정 가능한 plot 설정 ---
PLOT_CFG = {
    "figsize": (1.6, 1.2),
    "xlabel_fmt": r"$A_{{\tilde{{sp}}}}^{{({head})}}$",
    "ylabel": "Density",
    "alpha": 0.7,
    "dpi": 300,
    "ext": ".pdf",
}


def load(cfg: dict) -> pd.DataFrame:
    """지정된 (timestep, layer, head)의 데이터를 로드한다."""
    cols = ["prompt_idx", "timestep", "layer", "head", METRIC]
    dfs = []
    for fname, group in [(cfg["memorized_csv"], "Mem"), (cfg["unmemorized_csv"], "Non-Mem")]:
        df = pd.read_csv(RESULTS_DIR / fname, usecols=cols)
        df["group"] = group
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df[(df["timestep"] == cfg["timestep"]) & (df["layer"] == cfg["layer"]) & (df["head"] == cfg["head"])]


def plot(df: pd.DataFrame, cfg: dict) -> None:
    """density plot을 생성한다."""
    fig, ax = plt.subplots(figsize=PLOT_CFG["figsize"], constrained_layout=True)
    fig.get_layout_engine().set(w_pad=1 / 72, h_pad=1 / 72, wspace=0, hspace=0)

    for group, color in GROUP_PALETTE.items():
        values = df[df["group"] == group][METRIC]
        sns.kdeplot(values, ax=ax, color=color, label=group, fill=True, alpha=PLOT_CFG["alpha"], linewidth=0.75)

    ax.set_xlabel(PLOT_CFG["xlabel_fmt"].format(head=cfg["head"]), fontsize=7)
    ax.set_ylabel(PLOT_CFG["ylabel"], fontsize=7)
    ax.tick_params(labelsize=5)
    ax.legend(fontsize=5, loc="upper right")

    out = RESULTS_DIR / cfg["out"]
    fig.savefig(out)
    plt.close(fig)
    print(f"저장: {out}")


def main() -> None:
    for cfg in CONFIGS:
        df = load(cfg)
        plot(df, cfg)


if __name__ == "__main__":
    main()
