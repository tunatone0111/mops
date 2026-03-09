"""EOS cosine similarity vs mitigation score (조화평균 of CLIP score & SSCD) scatter plot."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

eos = pd.read_csv("results/mitigation/eos_similarity.csv")
met = pd.read_csv("results/mitigation/metrics.csv")

df = pd.merge(eos, met, on=["prompt_idx", "paraphrase_idx"])
x = df["eos_cosine_similarity"]


def harmonic_mean(a, b):
    """두 값의 조화평균. 둘 중 하나가 0 이하이면 0 반환."""
    return np.where((a > 0) & (b > 0), 2 * a * b / (a + b), 0.0)


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x_line = np.linspace(x.min(), x.max(), 100)

# (1) Mitigation score (mitigation)
y1 = harmonic_mean(df["clip_score_mitigation"].values, df["sscd_original_vs_mitigation"].values)
r1, p1 = stats.pearsonr(x, y1)
axes[0].scatter(x, y1, alpha=0.15, s=10, c="steelblue", edgecolors="none")
slope1, intercept1 = np.polyfit(x, y1, 1)
axes[0].plot(x_line, slope1 * x_line + intercept1, color="tomato", linewidth=2)
axes[0].set_xlabel("EOS Cosine Similarity (original vs paraphrase)", fontsize=12)
axes[0].set_ylabel("Mitigation Score (H-mean of CLIP & SSCD)", fontsize=12)
axes[0].set_title(f"EOS Similarity vs Mitigation Score (Masking)\n(r = {r1:.3f}, p = {p1:.2e}, n = {len(df)})", fontsize=13)
axes[0].grid(True, alpha=0.3)

# (2) Mitigation score (switching)
y2 = harmonic_mean(df["clip_score_switching"].values, df["sscd_original_vs_switching"].values)
r2, p2 = stats.pearsonr(x, y2)
axes[1].scatter(x, y2, alpha=0.15, s=10, c="darkorange", edgecolors="none")
slope2, intercept2 = np.polyfit(x, y2, 1)
axes[1].plot(x_line, slope2 * x_line + intercept2, color="tomato", linewidth=2)
axes[1].set_xlabel("EOS Cosine Similarity (original vs paraphrase)", fontsize=12)
axes[1].set_ylabel("Mitigation Score (H-mean of CLIP & SSCD)", fontsize=12)
axes[1].set_title(f"EOS Similarity vs Mitigation Score (Switching)\n(r = {r2:.3f}, p = {p2:.2e}, n = {len(df)})", fontsize=13)
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("results/mitigation/eos_vs_mitigation_score.png", dpi=150)
print(f"저장 완료: results/mitigation/eos_vs_mitigation_score.png")
print(f"Masking  - Pearson r = {r1:.4f}, p = {p1:.2e}, mean = {y1.mean():.4f}")
print(f"Switching - Pearson r = {r2:.4f}, p = {p2:.2e}, mean = {y2.mean():.4f}")
