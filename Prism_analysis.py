"""
==============================================================
 Prism Analysis — Step Selection Diagnostics & Visualization
==============================================================

Loads a Prism samples JSONL file and produces a suite of
diagnostic plots and statistics to understand *what* the
sampler selected and *why*.

Outputs:
  1. kld_distribution.png  — histogram of KLD values across
     all selected steps (global and per-problem)
  2. step_position_heatmap.png — heatmap of where in the
     reasoning chain selected steps tend to appear
  3. token_type_analysis.png — bar chart of token categories
     among selected tokens (numerals, operators, words, etc.)
  4. kld_vs_weight.png — scatter: per-step KLD vs. contrastive
     weight magnitude
  5. stats.json — summary statistics

Usage:
------
  python Prism_analysis.py --samples_path ./prism_samples.jsonl
  python Prism_analysis.py --samples_path ./PrismSamples/LR_Qwen1.5_gsm8k

==============================================================
"""

import json
import os
import re
import argparse
import collections
import numpy as np

# Matplotlib is used for all plotting. Install if needed:
#   pip install matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")   # headless-friendly backend
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found. Install with `pip install matplotlib` to enable plots.")


# =============================================================
# Token type classifier
# =============================================================
def classify_token(token_str: str) -> str:
    """Classify a decoded token string into a broad category."""
    t = token_str.strip()
    if not t:
        return "whitespace/punct"
    if re.fullmatch(r"[\d,.]+", t):
        return "numeral"
    if re.fullmatch(r"[+\-*/=<>^%()[\]{}|]", t):
        return "operator"
    if re.fullmatch(r"[a-zA-Z]+", t):
        return "word"
    if "\n" in token_str:
        return "newline"
    if re.fullmatch(r"[^\w\s]", t):
        return "symbol"
    return "mixed/other"


# =============================================================
# Load samples
# =============================================================
def load_samples(path: str):
    """Load JSONL samples from a file or directory."""
    samples = []
    if os.path.isdir(path):
        # Try to find a JSONL inside
        for fname in os.listdir(path):
            if fname.endswith(".jsonl"):
                path = os.path.join(path, fname)
                break
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    return samples


# =============================================================
# Analysis functions
# =============================================================
def kld_distribution(samples, output_dir):
    klds = [s["kl_divergence"] for s in samples if "kl_divergence" in s]
    stats = {
        "count": len(klds),
        "mean":  float(np.mean(klds)),
        "median": float(np.median(klds)),
        "std":   float(np.std(klds)),
        "p25":   float(np.percentile(klds, 25)),
        "p75":   float(np.percentile(klds, 75)),
        "p90":   float(np.percentile(klds, 90)),
        "max":   float(np.max(klds)),
    }
    print(f"\n[KLD Distribution]")
    for k, v in stats.items():
        print(f"  {k:<10}: {v}")

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(klds, bins=60, color="#5A7FD4", edgecolor="none", alpha=0.85)
        ax.axvline(stats["mean"], color="#E05252", linewidth=1.5, linestyle="--", label=f"mean={stats['mean']:.2f}")
        ax.axvline(stats["median"], color="#52C0E0", linewidth=1.5, linestyle=":", label=f"median={stats['median']:.2f}")
        ax.set_xlabel("KL Divergence (Expert || Amateur)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Distribution of KLD at Selected Steps", fontsize=13)
        ax.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, "kld_distribution.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  → {path}")
    return stats


def step_position_analysis(samples, output_dir):
    """Analyse where in the reasoning chain selected steps appear."""
    # Group by prompt_id to get total chain lengths
    prompt_steps = collections.defaultdict(list)
    for s in samples:
        prompt_steps[s["prompt_id"]].append(s["step"])

    # Relative positions: step / max_step for that prompt
    rel_positions = []
    for pid, steps in prompt_steps.items():
        max_step = max(steps)
        for st in steps:
            rel_positions.append(st / max_step if max_step > 0 else 0.0)

    print(f"\n[Step Position]")
    print(f"  mean relative position: {np.mean(rel_positions):.3f}")
    print(f"  Selected steps span full chain uniformly? (std={np.std(rel_positions):.3f})")

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(rel_positions, bins=20, color="#7EC8A0", edgecolor="none", alpha=0.85)
        ax.set_xlabel("Relative Step Position (0=start, 1=end)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Where in the Reasoning Chain Are Steps Selected?", fontsize=13)
        plt.tight_layout()
        path = os.path.join(output_dir, "step_position_heatmap.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  → {path}")

    return {"mean_rel_pos": float(np.mean(rel_positions)), "std_rel_pos": float(np.std(rel_positions))}


def token_type_analysis(samples, output_dir):
    """Analyse what types of tokens dominate the selected vocabulary."""
    category_counts = collections.Counter()
    total_weighted  = collections.defaultdict(float)

    for s in samples:
        for tok_str, weight in zip(
            [str(tid) for tid in s.get("token_ids", [])],
            s.get("weights", [])
        ):
            # We only have token_ids in the JSONL, not decoded strings.
            # Decode them using a simple heuristic on the id value.
            # For proper decoding, pass --tokenizer_path.
            # Here we use a numeric proxy.
            try:
                tid = int(tok_str)
                # Rough proxy: low IDs tend to be punctuation/special,
                # mid-range are common words, high are rare/domain tokens.
                if tid < 300:
                    cat = "special/punct"
                elif re.fullmatch(r"\d+", tok_str):
                    # token id looks numeric — check value range
                    if 48 <= tid <= 57:  # ASCII digits
                        cat = "numeral"
                    else:
                        cat = "word/subword"
                else:
                    cat = "word/subword"
            except Exception:
                cat = "other"
            category_counts[cat] += 1
            total_weighted[cat]  += float(weight)

    print(f"\n[Token Type Distribution]")
    for cat, cnt in category_counts.most_common():
        print(f"  {cat:<20}: {cnt} tokens, total weight={total_weighted[cat]:.3f}")

    if HAS_MPL and category_counts:
        labels  = list(category_counts.keys())
        counts  = [category_counts[l] for l in labels]
        colors  = ["#5A7FD4", "#7EC8A0", "#E0B252", "#E05252", "#A07EC8"][:len(labels)]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, counts, color=colors, edgecolor="none", alpha=0.88)
        ax.set_xlabel("Token Category", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Token Type Distribution at Selected Steps", fontsize=13)
        plt.tight_layout()
        path = os.path.join(output_dir, "token_type_analysis.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  → {path}")

    return dict(category_counts)


def kld_vs_weight_scatter(samples, output_dir):
    """Scatter: KLD of a step vs. max contrastive weight at that step."""
    klds    = []
    max_wts = []
    for s in samples:
        if "kl_divergence" in s and s.get("weights"):
            klds.append(s["kl_divergence"])
            max_wts.append(max(s["weights"]))

    corr = float(np.corrcoef(klds, max_wts)[0, 1]) if len(klds) > 1 else 0.0
    print(f"\n[KLD vs. Max Weight] correlation = {corr:.4f}")

    if HAS_MPL and klds:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(klds, max_wts, alpha=0.25, s=8, color="#5A7FD4")
        ax.set_xlabel("KL Divergence (step)", fontsize=11)
        ax.set_ylabel("Max Contrastive Weight", fontsize=11)
        ax.set_title(f"KLD vs. Contrastive Weight (r={corr:.3f})", fontsize=13)
        plt.tight_layout()
        path = os.path.join(output_dir, "kld_vs_weight.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  → {path}")

    return {"correlation": corr, "n_points": len(klds)}


def per_problem_stats(samples):
    """Summary stats across problems."""
    prompt_counts = collections.Counter(s["prompt_id"] for s in samples)
    n_probs = len(prompt_counts)
    vals    = list(prompt_counts.values())
    print(f"\n[Per-Problem Stats]")
    print(f"  Problems covered  : {n_probs}")
    print(f"  Avg steps/problem : {np.mean(vals):.1f}")
    print(f"  Std steps/problem : {np.std(vals):.1f}")
    print(f"  Max steps/problem : {max(vals)}")
    print(f"  Min steps/problem : {min(vals)}")
    return {
        "n_problems": n_probs,
        "mean_steps_per_problem": float(np.mean(vals)),
        "std_steps_per_problem":  float(np.std(vals)),
    }


# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="Prism Analysis — Step Selection Diagnostics")
    parser.add_argument("--samples_path", type=str, required=True,
                        help="Path to prism_samples.jsonl (or directory containing one)")
    parser.add_argument("--output_dir", type=str, default="./prism_analysis_output",
                        help="Directory to save plots and stats")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading samples from: {args.samples_path}")
    samples = load_samples(args.samples_path)
    print(f"Loaded {len(samples)} samples.")

    if not samples:
        print("No samples found. Check the path.")
        return

    all_stats = {}

    all_stats["kld"]          = kld_distribution(samples, args.output_dir)
    all_stats["step_position"] = step_position_analysis(samples, args.output_dir)
    all_stats["token_types"]  = token_type_analysis(samples, args.output_dir)
    all_stats["kld_vs_weight"] = kld_vs_weight_scatter(samples, args.output_dir)
    all_stats["per_problem"]  = per_problem_stats(samples)

    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nAll analysis complete. Outputs in: {args.output_dir}")
    if not HAS_MPL:
        print("Install matplotlib (`pip install matplotlib`) to generate plots.")


if __name__ == "__main__":
    main()
