"""
Generate comparison chart for memory decay demo.

Run: uv run python generate_chart.py
Output: memory-decay-comparison.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RED = "#F28B82"
GREEN = "#5FBFA4"
BLUE = "#7BAAF7"

tests = [
    "Test 1\nStateless\n(no memory)",
    "Test 2\nStateful\n(agent.state)",
    "Test 3\nPersistent\n(FileSessionManager)",
]

# Three capability dimensions scored 0 or 1
remembers = [0, 1, 1]
cross_session = [0, 0, 1]
personalized = [0, 1, 1]

x = np.arange(len(tests))
width = 0.22

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars1 = ax.bar(x - width, remembers, width, label="Remembers preferences", color=RED, zorder=3)
bars2 = ax.bar(x, cross_session, width, label="Cross-session persistence", color=BLUE, zorder=3)
bars3 = ax.bar(x + width, personalized, width, label="Personalized results", color=GREEN, zorder=3)

# Labels on bars
for bars, values in [(bars1, remembers), (bars2, cross_session), (bars3, personalized)]:
    for bar, val in zip(bars, values):
        label = "Yes" if val == 1 else "No"
        color = "#333333" if val == 1 else "#999999"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            label,
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=color,
        )

ax.set_title("Agent Memory Capabilities by Approach", fontsize=16, fontweight="bold", pad=16)
ax.set_ylabel("Capability", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(tests, fontsize=11)
ax.set_ylim(0, 1.35)
ax.set_yticks([0, 1])
ax.set_yticklabels(["No", "Yes"], fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

ax.text(
    0.5, -0.15,
    "Same 3-turn travel conversation: search Tokyo → book ryokan → search Zurich. Only the memory approach changes.",
    transform=ax.transAxes, ha="center", fontsize=9, color="#666666",
)

plt.tight_layout()
plt.savefig("memory-decay-comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: memory-decay-comparison.png")
