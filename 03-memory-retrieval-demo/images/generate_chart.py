"""
Generate token comparison chart for memory retrieval demo.

Run: uv run python generate_chart.py
Output: memory-retrieval-tokens.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RED = "#F28B82"
ORANGE = "#FFB347"
GREEN = "#5FBFA4"

strategies = [
    "Dump All\n(8 sections)",
    "Keyword\nSearch",
    "Semantic Search\n(top-3)",
]

# Token counts from actual test runs
tokens = [1127, 109, 436]
colors = [RED, ORANGE, GREEN]

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars = ax.bar(strategies, tokens, color=colors, width=0.5, zorder=3)

# Value labels on bars
for bar, value in zip(bars, tokens):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 20,
        f"{value:,}",
        ha="center", va="bottom",
        fontsize=14, fontweight="bold",
    )

ax.set_title("Token Usage by Memory Retrieval Strategy", fontsize=16, fontweight="bold", pad=16)
ax.set_ylabel("Tokens in Context", fontsize=12)
ax.set_ylim(0, 1400)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

legend_patches = [
    mpatches.Patch(color=RED, label="Dump all — loads entire memory (inefficient)"),
    mpatches.Patch(color=ORANGE, label="Keyword — exact match (missed relevant sections)"),
    mpatches.Patch(color=GREEN, label="Semantic — top-3 relevant (accurate + efficient)"),
]
ax.legend(handles=legend_patches, loc="upper right", fontsize=10, framealpha=0.9)

# Annotation: savings
ax.annotate(
    "61% fewer\ntokens",
    xy=(2, 436), xytext=(1.5, 900),
    fontsize=12, fontweight="bold", color=GREEN,
    ha="center",
    arrowprops=dict(arrowstyle="->", color=GREEN, lw=2),
)

ax.text(
    0.5, -0.12,
    'Query: "What food do I like?" — 8 memory sections (persona, travel, food, work, trips, loyalty, comms, emergency).',
    transform=ax.transAxes, ha="center", fontsize=9, color="#666666",
)

plt.tight_layout()
plt.savefig("memory-retrieval-tokens.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: memory-retrieval-tokens.png")
