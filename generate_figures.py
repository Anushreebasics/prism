"""
Generate all figures for the extended Prism paper.
Saves to ./assets/ directory.
Run: python generate_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

os.makedirs("./assets", exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
COLORS = {
    "base":     "#8E9BAE",
    "sft":      "#5B8DB8",
    "prism":    "#E07B39",
    "iprism2":  "#D45A20",
    "iprism3":  "#B03A10",
    "entropy":  "#7CB87C",
    "ensemble": "#9B7CC8",
    "bg":       "#FAFAFA",
    "grid":     "#E8E8E8",
}
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.facecolor":  COLORS["bg"],
    "figure.facecolor": "white",
    "axes.grid":       True,
    "grid.color":      COLORS["grid"],
    "grid.linewidth":  0.8,
})


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: iPrism Round-by-Round Progression
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.2))

rounds  = [0, 1, 2, 3]
gsm8k   = [42.5, 70.6, 72.1, 73.0]
math    = [34.2, 59.3, 61.4, 62.0]
svamp   = [68.8, 76.0, 77.3, 77.9]
asdiv   = [68.1, 79.8, 81.2, 81.7]

ax.plot(rounds, gsm8k,  "o-", color=COLORS["prism"],   lw=2.2, ms=7, label="GSM8K")
ax.plot(rounds, math,   "s-", color=COLORS["sft"],     lw=2.2, ms=7, label="MATH")
ax.plot(rounds, svamp,  "^-", color=COLORS["entropy"],  lw=2.2, ms=7, label="SVAMP")
ax.plot(rounds, asdiv,  "D-", color=COLORS["ensemble"], lw=2.2, ms=7, label="ASDiv")

# Annotate SFT baselines as dashed horizontal lines
ax.axhline(69.2, color=COLORS["prism"],   lw=1.0, ls="--", alpha=0.45)
ax.axhline(57.1, color=COLORS["sft"],     lw=1.0, ls="--", alpha=0.45)
ax.text(3.08, 69.6, "SFT", fontsize=8.5, color=COLORS["prism"],   alpha=0.7)
ax.text(3.08, 57.5, "SFT", fontsize=8.5, color=COLORS["sft"],     alpha=0.7)

ax.set_xticks(rounds)
ax.set_xticklabels(["Round 0\n(base)", "Round 1\n(Prism)", "Round 2\n(iPrism)", "Round 3\n(iPrism)"])
ax.set_ylabel("Zero-shot Pass@1 Accuracy (%)")
ax.set_title("iPrism: Round-by-Round Improvement (Qwen2.5-Math-1.5B)", fontsize=12, fontweight="bold", pad=10)
ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)
ax.set_ylim(28, 88)
plt.tight_layout()
plt.savefig("./assets/iprism_rounds.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ assets/iprism_rounds.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Cross-Domain Results
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

# Code domain
benchmarks_code  = ["HumanEval", "MBPP"]
base_code  = [51.2, 48.6]
sft_code   = [56.8, 53.2]
prism_code = [59.1, 55.7]

x = np.arange(len(benchmarks_code))
w = 0.24
ax = axes[0]
ax.bar(x - w, base_code,  w, label="Base",  color=COLORS["base"],  edgecolor="white")
ax.bar(x,     sft_code,   w, label="+ SFT", color=COLORS["sft"],   edgecolor="white")
ax.bar(x + w, prism_code, w, label="+ Prism", color=COLORS["prism"], edgecolor="white")
for xi, (b, s, p) in enumerate(zip(base_code, sft_code, prism_code)):
    ax.text(xi + w, p + 0.5, f"{p:.1f}", ha="center", va="bottom", fontsize=8.5, color=COLORS["prism"], fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(benchmarks_code)
ax.set_ylabel("Pass@1 Accuracy (%)")
ax.set_title("Code Generation\n(DeepSeek-Coder-1.3B-Instruct)", fontsize=10.5, fontweight="bold")
ax.set_ylim(40, 66)
ax.legend(fontsize=9)

# Logic domain
benchmarks_logic  = ["ARC-Challenge", "LogiQA"]
base_logic  = [62.4, 41.3]
sft_logic   = [65.1, 43.7]
prism_logic = [66.8, 45.2]

x = np.arange(len(benchmarks_logic))
ax = axes[1]
ax.bar(x - w, base_logic,  w, color=COLORS["base"],  edgecolor="white")
ax.bar(x,     sft_logic,   w, color=COLORS["sft"],   edgecolor="white")
ax.bar(x + w, prism_logic, w, color=COLORS["prism"], edgecolor="white")
for xi, (b, s, p) in enumerate(zip(base_logic, sft_logic, prism_logic)):
    ax.text(xi + w, p + 0.3, f"{p:.1f}", ha="center", va="bottom", fontsize=8.5, color=COLORS["prism"], fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(benchmarks_logic)
ax.set_title("Logical Reasoning\n(Qwen2.5-7B-Instruct)", fontsize=10.5, fontweight="bold")
ax.set_ylim(35, 72)

fig.suptitle("Prism Cross-Domain Generalization (no hyperparameter changes)", fontsize=11.5, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("./assets/cross_domain.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ assets/cross_domain.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Extensions Cumulative Improvement (bar chart)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

methods = ["SFT", "Prism\n(R=1)", "Prism +\nEntropy Filter", "Prism +\nEnsemble (N=3)", "iPrism\n(R=3)"]
gsm = [69.2, 70.6, 71.8, 72.3, 73.0]
mat = [57.1, 59.3, 60.8, 60.9, 62.0]
palette = [COLORS["sft"], COLORS["prism"], COLORS["entropy"], COLORS["ensemble"], COLORS["iprism3"]]

for ax, scores, title in zip(axes, [gsm, mat], ["GSM8K", "MATH"]):
    bars = ax.bar(range(len(methods)), scores, color=palette, edgecolor="white", width=0.62)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=8.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{title} Accuracy", fontsize=11, fontweight="bold")
    lo = min(scores) - 3
    ax.set_ylim(lo, max(scores) + 3)

fig.suptitle("Cumulative Gains: Prism Extensions vs. SFT (Qwen2.5-Math-1.5B)", fontsize=11.5, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("./assets/extensions_comparison.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ assets/extensions_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Entropy Filter Ablation
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8, 4.0))

labels = ["No Filter\n(100% steps)", "Fixed γ=2.0\n(~84% steps)", "Adaptive γ\n(~87% steps)"]
gsm_ef = [70.6, 71.2, 71.8]
mat_ef = [59.3, 60.1, 60.8]
colors_ef = [COLORS["base"], COLORS["sft"], COLORS["entropy"]]

for ax, scores, title in zip(axes, [gsm_ef, mat_ef], ["GSM8K", "MATH"]):
    bars = ax.bar(range(3), scores, color=colors_ef, edgecolor="white", width=0.5)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f"{title}", fontsize=11, fontweight="bold")
    ax.set_ylim(min(scores) - 2, max(scores) + 2)
    ax.set_ylabel("Accuracy (%)")

fig.suptitle("Effect of Entropy Filter (γ) on Step Selection Quality", fontsize=11.5, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("./assets/entropy_filter_ablation.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ assets/entropy_filter_ablation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Ensemble Size Ablation
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.0))

n_amateurs = [1, 2, 3]
gsm_ens = [70.6, 71.4, 72.3]
mat_ens = [59.3, 60.0, 60.9]

ax.plot(n_amateurs, gsm_ens, "o-", color=COLORS["prism"],   lw=2.2, ms=8, label="GSM8K")
ax.plot(n_amateurs, mat_ens, "s-", color=COLORS["ensemble"], lw=2.2, ms=8, label="MATH")

for n, g, m in zip(n_amateurs, gsm_ens, mat_ens):
    ax.annotate(f"{g:.1f}", (n, g), textcoords="offset points", xytext=(6, 4),
                fontsize=9.5, color=COLORS["prism"], fontweight="bold")
    ax.annotate(f"{m:.1f}", (n, m), textcoords="offset points", xytext=(6, -12),
                fontsize=9.5, color=COLORS["ensemble"], fontweight="bold")

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["N=1\n(Qwen2.5-0.5B)", "N=2\n(+ Qwen2.5-1.5B)", "N=3\n(+ GPT-2)"])
ax.set_ylabel("Accuracy (%)")
ax.set_title("Multi-Amateur Ensemble: Effect of N", fontsize=11.5, fontweight="bold", pad=10)
ax.legend(fontsize=10)
ax.set_ylim(56, 76)
plt.tight_layout()
plt.savefig("./assets/ensemble_size.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ assets/ensemble_size.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: KLD Step Selection — Conceptual Illustration
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 4.5)
ax.axis("off")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# Mock KLD trace across steps
np.random.seed(42)
steps   = np.linspace(0, 100, 120)
kld_val = np.abs(np.sin(steps * 0.18) * 2.5 + np.random.randn(120) * 0.5 + 1.2)
kld_val = np.clip(kld_val, 0, 5)

# Draw the KLD trace
ax.plot(steps, kld_val, color="#8E9BAE", lw=1.4, alpha=0.85, zorder=2)

beta = 2.0
gamma_entropy = 2.8  # conceptual: entropy ceiling line (inverted logic shown as band)

ax.axhline(beta, color=COLORS["prism"], lw=1.4, ls="--", alpha=0.9, label=f"KLD threshold (β={beta})")

# Shade selected regions (KLD > beta)
mask = kld_val > beta
for i in range(len(steps) - 1):
    if mask[i]:
        ax.axvspan(steps[i], steps[i+1], alpha=0.18, color=COLORS["prism"], zorder=1)

# Annotate categories
ax.text(12, 0.6, "Easy steps\n(KLD < β)\n→ skipped", ha="center", fontsize=8.5,
        color="#666", style="italic")
ax.text(50, 3.8, "Hard steps\n(KLD > β)\n→ SELECTED", ha="center", fontsize=8.5,
        color=COLORS["prism"], fontweight="bold")
ax.text(82, 0.4, "Easy steps\n→ skipped", ha="center", fontsize=8.5,
        color="#666", style="italic")

ax.set_xlabel("Step position in reasoning chain", fontsize=10, labelpad=4)
ax.set_ylabel("KL Divergence\n(Expert ∥ Amateur)", fontsize=10)
ax.set_title("Prism Step Selection: KLD-Driven Informative Step Identification", fontsize=11.5, fontweight="bold", pad=10)
ax.set_ylim(-0.2, 5.2)
ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
ax.spines["bottom"].set_visible(True)
ax.spines["left"].set_visible(True)
ax.axis("on")
ax.set_facecolor(COLORS["bg"])
ax.grid(True, color=COLORS["grid"], linewidth=0.8)
plt.tight_layout()
plt.savefig("./assets/kld_selection_illustration.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ assets/kld_selection_illustration.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: iPrism Efficiency vs Accuracy (scatter)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

methods_eff = ["SFT", "Prism\n(R=1)", "iPrism\n(R=2)", "iPrism\n(R=3)"]
time_h      = [4.0,  0.5,  1.0,  1.5]
gsm_eff     = [69.2, 70.6, 72.1, 73.0]
sizes       = [180,  120,  150,  180]
colors_eff  = [COLORS["sft"], COLORS["prism"], COLORS["iprism2"], COLORS["iprism3"]]

for i, (t, g, s, c, label) in enumerate(zip(time_h, gsm_eff, sizes, colors_eff, methods_eff)):
    ax.scatter(t, g, s=s, color=c, edgecolors="white", linewidths=1.5, zorder=3)
    ax.annotate(label, (t, g), textcoords="offset points",
                xytext=(8, 4), fontsize=9, color=c, fontweight="bold")

ax.set_xlabel("Total Wall-Clock Time (hours, single H200 GPU)", fontsize=10.5)
ax.set_ylabel("GSM8K Accuracy (%)", fontsize=10.5)
ax.set_title("Efficiency vs. Accuracy: SFT vs. Prism Variants", fontsize=12, fontweight="bold", pad=10)
ax.set_xlim(-0.3, 5.0)
ax.set_ylim(66, 76)

# Preferred region annotation
ax.annotate("", xy=(1.6, 73.5), xytext=(0.2, 73.5),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
ax.text(1.7, 73.5, "Better →", fontsize=8.5, color="#555", va="center")

plt.tight_layout()
plt.savefig("./assets/efficiency_vs_accuracy.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ assets/efficiency_vs_accuracy.png")

print("\nAll figures generated in ./assets/")
