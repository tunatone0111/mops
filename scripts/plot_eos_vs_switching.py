"""EOS cosine similarity vs switching 메트릭(SSCD, CLIP score) scatter plot."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

eos = pd.read_csv("results/mitigation/eos_similarity.csv")
met = pd.read_csv("results/mitigation/metrics.csv")

df = pd.merge(eos, met, on=["prompt_idx", "paraphrase_idx"])
x = df["eos_cosine_similarity"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (1) EOS vs SSCD (orig vs switching)
y1 = df["sscd_original_vs_switching"]
r1, p1 = stats.pearsonr(x, y1)
axes[0].scatter(x, y1, alpha=0.15, s=10, c="steelblue", edgecolors="none")
slope1, intercept1 = np.polyfit(x, y1, 1)
x_line = np.linspace(x.min(), x.max(), 100)
axes[0].plot(x_line, slope1 * x_line + intercept1, color="tomato", linewidth=2)
axes[0].set_xlabel("EOS Cosine Similarity (original vs paraphrase)", fontsize=12)
axes[0].set_ylabel("SSCD (original vs switching)", fontsize=12)
axes[0].set_title(f"EOS Similarity vs SSCD (Switching)\n(r = {r1:.3f}, p = {p1:.2e}, n = {len(df)})", fontsize=13)
axes[0].grid(True, alpha=0.3)

# (2) EOS vs CLIP score (switching)
y2 = df["clip_score_switching"]
r2, p2 = stats.pearsonr(x, y2)
axes[1].scatter(x, y2, alpha=0.15, s=10, c="darkorange", edgecolors="none")
slope2, intercept2 = np.polyfit(x, y2, 1)
axes[1].plot(x_line, slope2 * x_line + intercept2, color="tomato", linewidth=2)
axes[1].set_xlabel("EOS Cosine Similarity (original vs paraphrase)", fontsize=12)
axes[1].set_ylabel("CLIP Score (switching vs original prompt)", fontsize=12)
axes[1].set_title(f"EOS Similarity vs CLIP Score (Switching)\n(r = {r2:.3f}, p = {p2:.2e}, n = {len(df)})", fontsize=13)
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("results/mitigation/eos_vs_switching.png", dpi=150)
print(f"저장 완료: results/mitigation/eos_vs_switching.png")
print(f"EOS vs SSCD(switching): Pearson r = {r1:.4f}, p = {p1:.2e}")
print(f"EOS vs CLIP(switching): Pearson r = {r2:.4f}, p = {p2:.2e}")
