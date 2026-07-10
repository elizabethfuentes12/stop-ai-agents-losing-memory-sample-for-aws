"""
Generate core memory evolution chart for core memory demo.

Run: uv run python generate_chart.py
Output: core-memory-evolution.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RED = "#F28B82"
GREEN = "#5FBFA4"
BLUE = "#7BAAF7"
ORANGE = "#FFB347"
YELLOW = "#FFD966"

tests = [
    "Test 1\nNo memory\ntools",
    "Test 2\nCore memory\ntools",
    "Test 3\nMemory\nevolution",
    "Test 4\nCross-session\npersistence",
]

# Metrics from actual test runs
sections_stored = [0, 2, 2, 2]
memory_versions = [0, 2, 3, 2]
colors_sections = [RED, GREEN, GREEN, GREEN]
colors_versions = [RED, BLUE, ORANGE, BLUE]

x = np.arange(len(tests))
width = 0.3

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars1 = ax.bar(x - width / 2, sections_stored, width, label="Memory sections stored", color=colors_sections, zorder=3)
bars2 = ax.bar(x + width / 2, memory_versions, width, label="Total version updates", color=colors_versions, zorder=3)

# Value labels
for bars, values in [(bars1, sections_stored), (bars2, memory_versions)]:
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                str(val),
                ha="center", va="bottom",
                fontsize=13, fontweight="bold",
            )

ax.set_title("Core Memory: Sections Stored and Version Updates", fontsize=16, fontweight="bold", pad=16)
ax.set_ylabel("Count", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(tests, fontsize=10)
ax.set_ylim(0, 4.5)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

legend_patches = [
    mpatches.Patch(color=RED, label="No memory / empty"),
    mpatches.Patch(color=GREEN, label="Sections stored (persona, preferences)"),
    mpatches.Patch(color=BLUE, label="Version updates (write + update ops)"),
    mpatches.Patch(color=ORANGE, label="Evolution (preferences changed)"),
]
ax.legend(handles=legend_patches, loc="upper left", fontsize=9, framealpha=0.9)

ax.text(
    0.5, -0.15,
    "Agent autonomously creates and updates memory sections. Version tracking shows memory evolution over time.",
    transform=ax.transAxes, ha="center", fontsize=9, color="#666666",
)

plt.tight_layout()
plt.savefig("core-memory-evolution.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: core-memory-evolution.png")
