"""Draws the report's charts from eval_P5_T50.md, so a chart can never disagree with the evaluation file.

python tools/report_charts.py          (needs matplotlib; writes docs/img/chart-*.png)
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL = (ROOT / "eval_P5_T50.md").read_text(encoding="utf-8")
OUT = ROOT / "docs" / "img"

INK, SOFT, GRID = "#1c2733", "#4d5b69", "#dfe5ea"
BLUE, ORANGE = "#2f5d8a", "#c9500f"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 11, "text.color": INK, "axes.labelcolor": SOFT, "xtick.color": SOFT, "ytick.color": SOFT,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 200, "savefig.bbox": "tight", "axes.titlesize": 11.5, "axes.titleweight": "bold",
})


def rows(pattern: str) -> list[tuple[str, ...]]:
    return re.findall(pattern, EVAL, re.M)


def approaches() -> None:
    top1 = dict(rows(r"^\| (Classical|Deep learning|Hybrid) \|.*?\| ([\d.]+)% \|"))
    labels = [("Classical", "Classical features only"), ("Deep learning", "Network on the whole photo"), ("Hybrid", "Network on the vehicle crop")]
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    names, values = [n for _, n in labels][::-1], [float(top1[k]) for k, _ in labels][::-1]
    colours = [BLUE, "#8aa3bb", "#8aa3bb"]
    ax.barh(names, values, color=colours, height=0.55)
    for y, v in enumerate(values):
        ax.text(v + 1, y, f"{v:.2f}%", va="center", fontweight="bold")
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Top-1 accuracy on the test split (%)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    fig.savefig(OUT / "chart-approaches.png")
    plt.close(fig)


def robustness() -> None:
    data = rows(r"^\| (dark|haze|blur|low_res|occlusion) \| (\d) \| ([\d.]+)% \|$")
    clean = float(re.search(r"^\| clean \| 0 \| ([\d.]+)% \|$", EVAL, re.M).group(1))
    panels = [("dark", "Darkness"), ("haze", "Smoke or haze"), ("blur", "Blur"), ("low_res", "Low resolution"), ("occlusion", "Occlusion")]
    fig, axes = plt.subplots(1, 5, figsize=(9.4, 2.5), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        ys = [clean] + [float(v) for c, _, v in data if c == key]
        ax.plot(range(4), ys, color=BLUE, linewidth=2, marker="o", markersize=5, markeredgecolor="white", markeredgewidth=1)
        ax.text(3, ys[-1] - 7, f"{ys[-1]:.1f}%", ha="right", va="top", fontweight="bold")
        ax.set_title(title)
        ax.set_xticks(range(4), ["None", "Mild", "Med", "Severe"], fontsize=8.5)
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Top-1 accuracy (%)")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "chart-robustness.png")
    plt.close(fig)


def thresholds() -> None:
    data = [(float(a), float(b), float(c)) for a, b, c in rows(r"^\| (0\.\d+) \| ([\d.]+)% \| ([\d.]+)% \|$")]
    argmax = float(re.search(r"\(argmax\) would miss ([\d.]+)%", EVAL).group(1))
    xs = [d[0] for d in data]
    fig, ax = plt.subplots(figsize=(7.2, 3))
    ax.axhline(argmax, color=SOFT, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(0.05, argmax + 0.1, f"Missed if we trust only the top answer: {argmax:.2f}%", ha="left", color=SOFT, fontsize=9.5)
    ax.plot(xs, [d[1] for d in data], color=ORANGE, linewidth=2, marker="o", markersize=6, markeredgecolor="white", label="Hybrids and EVs missed")
    ax.plot(xs, [d[2] for d in data], color=BLUE, linewidth=2, marker="s", markersize=6, markeredgecolor="white", label="False alarms")
    ax.text(xs[0] + 0.022, data[0][1] - 0.02, f"In use: {data[0][1]:.2f}% missed", va="center", fontweight="bold", fontsize=10)
    ax.set_xticks(xs, [f"{x:g}" for x in xs])
    ax.set_xlabel("Warning threshold on summed hybrid and EV probability")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 6.2)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(OUT / "chart-threshold.png")
    plt.close(fig)


def coverage() -> None:
    data = [(int(a), float(b)) for a, b in rows(r"^\| top (\d+)% of photos \| ([\d.]+)% \|$")]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.plot([d[0] for d in data], [d[1] for d in data], color=BLUE, linewidth=2, marker="o", markersize=6, markeredgecolor="white")
    for x, y in (data[0], data[2]):
        ax.annotate(f"{y:.2f}%", (x, y), xytext=(8, -14), textcoords="offset points", ha="left", fontweight="bold", fontsize=10)
    ax.invert_xaxis()
    ax.set_ylim(92, 100)
    ax.set_xlabel("Share of photos answered, most confident first (%)")
    ax.set_ylabel("Top-1 accuracy (%)")
    fig.savefig(OUT / "chart-coverage.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    approaches()
    robustness()
    thresholds()
    coverage()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("chart-*.png"))))
