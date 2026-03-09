"""EOS cosine similarity vs CLIP score (mitigation) scatter plot."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

eos = pd.read_csv("results/mitigation/eos_similarity.csv")
met = pd.read_csv("results/mitigation/metrics.csv")

df = pd.merge(eos, met, on=["prompt_idx", "paraphrase_idx"])

x = df["eos_cosine_similarity"]
y = df["clip_score_mitigation"]

r, p = stats.pearsonr(x, y)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, alpha=0.15, s=10, c="steelblue", edgecolors="none")

slope, intercept = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, slope * x_line + intercept, color="tomato", linewidth=2)

ax.set_xlabel("EOS Cosine Similarity (original vs paraphrase)", fontsize=12)
ax.set_ylabel("CLIP Score (mitigation image vs original prompt)", fontsize=12)
ax.set_title(f"EOS Similarity vs CLIP Score (Mitigation)\n(r = {r:.3f}, p = {p:.2e}, n = {len(df)})", fontsize=13)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("results/mitigation/eos_vs_clip_mitigation.png", dpi=150)
print(f"저장 완료: results/mitigation/eos_vs_clip_mitigation.png")
print(f"Pearson r = {r:.4f}, p = {p:.2e}")
