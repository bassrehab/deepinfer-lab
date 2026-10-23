#!/usr/bin/env python3
"""Render the feasibility figure from results/economics.json.

Left: GPUs implied by $38/GPU-hour versus GPUs needed to compute the misses and
outputs at two decode operating points from DeepInfer's own Pareto chart.
Right: SSD needed to hold seven days of computed tokens at the same points, against
the ~100 PiB DeepSeek says it runs.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "results/economics.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/feasibility_band.png"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated adjacent pairs, light surface
INK, MUTED, GRID = "#1a1a1a", "#6b6761", "#e6e2d9"

d = json.load(open(SRC))
base = d["base"]
hi = d["grid"]["128/rank 22.8K | 64K batch 89.5K"]
x = [100 * p["output_share"] for p in base]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=250)
fig.patch.set_facecolor("white")

# Left: GPU count
ax.plot(x, [p["gpus_from_revenue"] / 1e3 for p in base], color=BLUE, lw=2.4, zorder=3)
ax.plot(x, [p["gpus_from_capacity"] / 1e3 for p in base], color=ORANGE, lw=2, ls="--", zorder=3)
ax.plot(x, [p["gpus_from_capacity"] / 1e3 for p in hi], color=AQUA, lw=2, ls="--", zorder=3)
for xs, ys, c, label, dy in (
    (x, [p["gpus_from_revenue"] / 1e3 for p in base], BLUE, "implied by $38 / GPU-hour", 0.5),
    (x, [p["gpus_from_capacity"] / 1e3 for p in base], ORANGE, "needed at 60 req/rank (14.5K tok/s/GPU)", 0.9),
    (x, [p["gpus_from_capacity"] / 1e3 for p in hi], AQUA, "needed at 128 req/rank (22.8K tok/s/GPU)", -0.9),
):
    ax.annotate(label, (xs[-1], ys[-1] + dy), ha="right", va="bottom", fontsize=9.5, color=INK)
for p in base[1::2]:
    ax.annotate(f"hit {100*p['hit_rate']:.0f}%", (100 * p["output_share"], p["gpus_from_capacity"] / 1e3), xytext=(4, -12), textcoords="offset points", fontsize=8.5, color=MUTED)
for p in hi[1::2]:
    ax.annotate(f"{100*p['hit_rate']:.0f}%", (100 * p["output_share"], p["gpus_from_capacity"] / 1e3), xytext=(4, 4), textcoords="offset points", fontsize=8.5, color=MUTED)
ax.set_xlabel("output tokens as a share of all tokens served (%)")
ax.set_ylabel("GPUs (thousands)")
ax.set_title("How many GPUs the disclosed numbers imply", loc="left", fontsize=12, color=INK)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlim(0, 5.3); ax.set_ylim(0, 16.5)

# Right: storage at 7-day TTL
bx.plot(x, [p["storage"]["needed_pib_at_ttl"]["7d"] for p in base], color=ORANGE, lw=2, ls="--")
bx.plot(x, [p["storage"]["needed_pib_at_ttl"]["7d"] for p in hi], color=AQUA, lw=2, ls="--")
bx.axhline(100, color=MUTED, lw=1.2, ls=":")
bx.annotate("~100 PiB DeepSeek says it runs", (0.1, 103), fontsize=9.5, color=INK, va="bottom")
bx.annotate("60 req/rank", (x[-1], base[-1]["storage"]["needed_pib_at_ttl"]["7d"] + 6), ha="right", fontsize=9.5, color=INK)
bx.annotate("128 req/rank", (x[-1], hi[-1]["storage"]["needed_pib_at_ttl"]["7d"] - 18), ha="right", fontsize=9.5, color=INK)
bx.set_xlabel("output tokens as a share of all tokens served (%)")
bx.set_ylabel("SSD to hold 7 days of computed tokens (PiB)")
bx.set_title("Where the 100 PiB cache fits", loc="left", fontsize=12, color=INK)
bx.grid(axis="y", color=GRID, lw=0.8); bx.set_axisbelow(True)
bx.set_xlim(0, 5.3); bx.set_ylim(0, 340)

fig.text(0.01, 0.01, "Inputs: 155T tokens/day, $38/GPU-hour off-peak, deepseek-flash off-peak prices, DeepInfer Pareto charts, 80% online efficiency, 1:1 prefill:decode, 890 B/token, 1.2x erasure coding.",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT, facecolor="white")
print("wrote", OUT)
