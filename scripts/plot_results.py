"""Figures of the README, from the CSV files written by `qubo-alloc`.

Usage:  python3 scripts/plot_results.py results docs

Only reads what the C++ binary measured. Colours come from a validated
categorical palette (checked for colour-vision-deficiency separation);
lines also carry distinct markers and direct labels, so identity never
relies on colour alone.
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Fixed categorical order, one hue per solver everywhere in the README.
COLORS = {
    "sa": "#2a78d6",
    "sa-pair": "#eb6834",
    "sqa": "#1baf7a",
    "pt": "#eda100",
    "descent": "#e87ba4",
    "relax-round": "#6250d6",
    "equal-weight": "#8a8985",
}
MARKERS = {"sa": "o", "sa-pair": "s", "sqa": "D", "pt": "^", "descent": "v", "relax-round": "P", "equal-weight": "X"}
LABELS = {
    "sa": "SA (single flips)",
    "sa-pair": "SA + pair moves",
    "sqa": "SQA (path-integral)",
    "pt": "Parallel tempering",
    "descent": "Steepest descent",
    "relax-round": "Relax and round",
    "equal-weight": "Equal weight 1/N",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def landscape(res, out):
    rows = read(res / "landscape.csv")
    gaps = [float(r["gap_bp"]) for r in rows]
    walls = [float(r["min_wall_bp"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = [i * 50 for i in range(0, 50)]
    ax.hist(gaps, bins=bins, color=COLORS["sa"], label="energy of each feasible portfolio above the optimum")
    ax.hist(walls, bins=bins, color=COLORS["sa-pair"], label="lowest single-flip wall around it")
    ax.set_xlabel("basis points of annual utility")
    ax.set_ylabel("number of portfolios")
    ax.set_title(f"Every one of the {len(rows):,} feasible portfolios is a strict local minimum")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax.grid(axis="x", visible=False)
    fig.savefig(out / "landscape.png")
    plt.close(fig)


def tts(res, out):
    rows = read(res / "tts.csv")
    solvers = ["sa", "sa-pair", "sqa", "pt"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    for s in solvers:
        sel = [r for r in rows if r["solver"] == s and r["assets"] == "24"]
        xs = [int(r["sweeps"]) for r in sel]
        ys = [float(r["p_success"]) for r in sel]
        ax.plot(xs, ys, color=COLORS[s], marker=MARKERS[s], label=LABELS[s])
    ax.set_xscale("log")
    ax.set_xlabel("sweeps per run")
    ax.set_ylabel("P(run finds the exact optimum)")
    ax.set_title("Success per run, 24 assets, pick 6")
    ax.legend(loc="upper left")

    ax = axes[1]
    for s in solvers:
        pts = []
        for n in ["16", "20", "24"]:
            vals = [float(r["tts99"]) for r in rows if r["solver"] == s and r["assets"] == n and math.isfinite(float(r["tts99"]))]
            if vals:
                pts.append((int(n), min(vals) * 1e3))
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=COLORS[s], marker=MARKERS[s], label=LABELS[s])
        ax.annotate(LABELS[s], (xs[-1] + 0.3, ys[-1]), color=INK, fontsize=9, va="center")
    ax.set_yscale("log")
    ax.set_xlim(15, 30)
    ax.set_xticks([16, 20, 24])
    ax.set_xlabel("assets (variables)")
    ax.set_ylabel("best time to solution, 99% (ms, 1 core)")
    ax.set_title("Time to solution")
    fig.savefig(out / "tts.png")
    plt.close(fig)


def scaling(res, out):
    rows = read(res / "scaling.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, n in zip(axes, ["50", "100", "200"]):
        for s in ["descent", "sa", "pt", "sqa", "sa-pair"]:
            sel = [r for r in rows if r["solver"] == s and r["assets"] == n]
            xs = [float(r["seconds"]) * 1e3 for r in sel]
            ys = [max(float(r["mean_gap_bp"]), 0.05) for r in sel]  # 0 bp shown at the floor
            ax.plot(xs, ys, color=COLORS[s], marker=MARKERS[s], label=LABELS[s], ls="-" if len(xs) > 1 else "none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{n} assets, pick {int(n) // 5}")
        ax.set_xlabel("time per solve (ms, 1 core)")
    axes[0].set_ylabel("gap to best known (bp of utility)")
    axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Quality vs time at scale (points at 0.05 bp found the best known solution)", color=INK, fontweight="bold", y=1.03)
    fig.savefig(out / "scaling.png")
    plt.close(fig)


def backtest(res, out):
    rows = read(res / "backtest_metrics.csv")
    names = [r["method"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    y = list(range(len(names)))[::-1]

    ax = axes[0]
    sh = [float(r["sharpe"]) for r in rows]
    se = [float(r["sharpe_se"]) for r in rows]
    ax.barh(y, sh, xerr=[1.96 * e for e in se], color=[COLORS[n] for n in names], height=0.6, error_kw={"ecolor": INK_2, "lw": 1})
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[n] for n in names])
    ax.set_xlabel("Sharpe ratio (mean of 20 paths, 95% CI)")
    ax.set_title("Net Sharpe ratio")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    solv = [r for r in rows if r["method"] != "equal-weight"]
    ys = [y[names.index(r["method"])] for r in solv]
    ax.barh(ys, [max(float(r["obj_gap_bp"]), 0.1) for r in solv], color=[COLORS[r["method"]] for r in solv], height=0.6)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("mean gap to best solver on each date (bp)")
    ax.set_title("Optimisation quality")
    ax.grid(axis="y", visible=False)

    ax = axes[2]
    ax.barh(y, [float(r["turnover_per_year"]) for r in rows], color=[COLORS[n] for n in names], height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("turnover per year (x book)")
    ax.set_title("Trading")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Ten-year walk-forward, 40 assets, 20 independent markets, 10 bp costs", color=INK, fontweight="bold", y=1.02)
    fig.savefig(out / "backtest.png")
    plt.close(fig)

    # Equity curves of the first path.
    eq = read(res / "equity.csv")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    days = [int(r["day"]) / 252 for r in eq]
    ends = []
    for s in ["equal-weight", "relax-round", "sa-pair", "sqa", "sa"]:
        ys = [float(r[s]) for r in eq]
        ax.plot(days, ys, color=COLORS[s], lw=1.6, label=LABELS[s])
        ends.append([ys[-1], s])
    # Spread end labels so that close finishes do not print on top of each other.
    ends.sort()
    gap = 0.045 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
    for y_end, s in ends:
        ax.annotate(LABELS[s], (days[-1] + 0.15, y_end), color=INK, fontsize=9, va="center")
    ax.set_xlim(0, days[-1] + 2.2)
    ax.set_xlabel("years since first trade")
    ax.set_ylabel("net equity (start = 1)")
    ax.set_title("One simulated market (path 0): equity net of 10 bp costs")
    fig.savefig(out / "equity.png")
    plt.close(fig)


def paired_sharpe(res):
    """Paired differences vs equal weight: same market, so far less noise."""
    rows = read(res / "backtest_paths.csv")
    by = defaultdict(dict)
    for r in rows:
        by[r["method"]][r["path"]] = float(r["sharpe"])
    base = by["equal-weight"]
    print("Sharpe difference vs equal weight, paired over paths:")
    for m, d in by.items():
        if m == "equal-weight":
            continue
        diffs = [d[p] - base[p] for p in base]
        n = len(diffs)
        mean = sum(diffs) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1))
        print(f"  {m:<13} {mean:+.3f} +/- {sd / math.sqrt(n):.3f}  (t = {mean / (sd / math.sqrt(n)):+.2f})")


def perf(res, out):
    rows = read(res / "perf.csv")
    n = [int(r["variables"]) for r in rows]
    inc = [float(r["incremental_ns_per_proposal"]) for r in rows]
    naive = [float(r["naive_ns_per_proposal"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(n, naive, color=COLORS["sa"], marker="o", label="recompute E(x): O(n^2)")
    ax.plot(n, inc, color=COLORS["sa-pair"], marker="s", label="local fields: O(1) + O(n) on accept")
    ax.annotate("naive", (n[-1] * 1.08, naive[-1]), color=INK, fontsize=9, va="center")
    ax.annotate("incremental", (n[-1] * 1.08, inc[-1]), color=INK, fontsize=9, va="center")
    for x, a, b in zip(n, inc, naive):
        ax.annotate(f"{b / a:.0f}x", (x, math.sqrt(a * b)), color=INK_2, fontsize=8, ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(20, 700)
    ax.set_xticks(n)
    ax.set_xticklabels([str(v) for v in n])
    ax.minorticks_off()
    ax.set_xlabel("variables")
    ax.set_ylabel("ns per Metropolis proposal")
    ax.set_title("Incremental local fields vs naive energy")
    ax.legend(loc="upper left")
    fig.savefig(out / "perf.png")
    plt.close(fig)


def main():
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs")
    out.mkdir(parents=True, exist_ok=True)
    landscape(res, out)
    tts(res, out)
    scaling(res, out)
    backtest(res, out)
    perf(res, out)
    paired_sharpe(res)
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()

# -- laklaa.com
