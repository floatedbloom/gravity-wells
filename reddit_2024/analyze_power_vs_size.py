import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr
import matplotlib.pyplot as plt


def load_results(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    spearman_csv = root / "bias_vs_dogmatism_spearman.csv"
    kendall_csv = root / "bias_vs_dogmatism_kendall.csv"
    df_s = pd.read_csv(spearman_csv)
    df_k = pd.read_csv(kendall_csv)
    # Keep per-subreddit rows only
    df_s = df_s[df_s["scope"] == "per_subreddit"].copy()
    df_k = df_k[df_k["scope"] == "per_subreddit"].copy()
    # Clean
    df_s = df_s.replace([np.inf, -np.inf], np.nan).dropna(subset=["n", "spearman_rho", "p_value"]).copy()
    df_k = df_k.replace([np.inf, -np.inf], np.nan).dropna(subset=["n", "kendall_tau", "p_value"]).copy()
    # Ensure integer n
    df_s["n"] = df_s["n"].astype(int)
    df_k["n"] = df_k["n"].astype(int)
    return df_s, df_k


def safe_neglog10_p(p: pd.Series) -> pd.Series:
    p_clip = p.clip(lower=1e-300, upper=1.0)
    return -np.log10(p_clip)


def summarize_relationship(df: pd.DataFrame, effect_col: str, label: str) -> Dict[str, float]:
    results: Dict[str, float] = {}
    # Correlation between size and absolute effect
    abs_effect = df[effect_col].abs()
    r_e, p_e = pearsonr(df["n"], abs_effect)
    results["pearson_r_size_abs_effect_r"] = float(r_e)
    results["pearson_r_size_abs_effect_p"] = float(p_e)

    # Relationship between size and significance
    n_log = np.log10(df["n"].astype(float))
    sig = safe_neglog10_p(df["p_value"])  # larger is more significant
    slope, intercept, r, p, stderr = linregress(n_log, sig)
    results.update({
        "linreg_sig_vs_log10n_slope": float(slope),
        "linreg_sig_vs_log10n_intercept": float(intercept),
        "linreg_sig_vs_log10n_r": float(r),
        "linreg_sig_vs_log10n_r2": float(r * r),
        "linreg_sig_vs_log10n_p": float(p),
        "linreg_sig_vs_log10n_stderr": float(stderr),
    })
    return results


def make_plots(df_s: pd.DataFrame, df_k: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    # Spearman: size vs rho
    ax = axes[0, 0]
    ax.scatter(df_s["n"], df_s["spearman_rho"], alpha=0.8)
    ax.axhline(0, color="gray", lw=1)
    ax.set_xlabel("Subreddit size (n)")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Size vs effect (Spearman)")

    # Spearman: size vs significance (-log10 p)
    ax = axes[0, 1]
    sig_s = safe_neglog10_p(df_s["p_value"])
    ax.scatter(df_s["n"], sig_s, alpha=0.8)
    ax.set_xlabel("Subreddit size (n)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Size vs significance (Spearman)")

    # Kendall: size vs tau
    ax = axes[1, 0]
    ax.scatter(df_k["n"], df_k["kendall_tau"], alpha=0.8, color="#1f77b4")
    ax.axhline(0, color="gray", lw=1)
    ax.set_xlabel("Subreddit size (n)")
    ax.set_ylabel("Kendall tau-c")
    ax.set_title("Size vs effect (Kendall)")

    # Kendall: size vs significance (-log10 p)
    ax = axes[1, 1]
    sig_k = safe_neglog10_p(df_k["p_value"])
    ax.scatter(df_k["n"], sig_k, alpha=0.8, color="#1f77b4")
    ax.set_xlabel("Subreddit size (n)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Size vs significance (Kendall)")

    fig.suptitle("Subreddit size vs effect size and significance")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    df_s, df_k = load_results(root)

    # Summaries
    s_summary = summarize_relationship(df_s, effect_col="spearman_rho", label="Spearman")
    k_summary = summarize_relationship(df_k, effect_col="kendall_tau", label="Kendall")

    # Print concise discussion-oriented summary
    print("Power vs size summary (per-subreddit):")
    print("Spearman:")
    print({k: v for k, v in s_summary.items()})
    print("Kendall:")
    print({k: v for k, v in k_summary.items()})

    # Plots
    out_png = root / "out" / "power_vs_size.png"
    make_plots(df_s, df_k, out_png)
    print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    main()


