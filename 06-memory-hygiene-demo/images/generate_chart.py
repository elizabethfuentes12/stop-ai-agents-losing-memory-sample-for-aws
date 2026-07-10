"""
Generate the blast-radius chart for the memory hygiene demo.

Numbers come from test_memory_hygiene.py (measured on the live demo): how many of 4
questions/lookups are contaminated by ONE poisoned item, per backend and per defense state.
Deterministic checks against the store — no LLM judge, no invented numbers.

Run: uv run python generate_chart.py
Output: memory-hygiene-blast-radius.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RED = "#F28B82"      # poisoned (no defense)
GREEN = "#5FBFA4"    # gated (write-gate)
BLUE = "#7EA6E0"     # cleaned (forget)

TOTAL = 4
states = ["Poisoned\n(no defense)", "Gated\n(write-gate)", "Cleaned\n(forget)"]

# From test_memory_hygiene.py — contaminated answers of 4, per backend.
kv_values = [1, 0, 0]      # key-value: poison is one blob → blast radius 1
graph_values = [4, 0, 0]   # graph: poison propagates through every multi-hop traversal

x = np.arange(len(states))
width = 0.36

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars_kv = ax.bar(x - width / 2, kv_values, width, label="Key-value (agent.state)",
                 color="#B0B7C3", zorder=3)
bars_graph = ax.bar(x + width / 2, graph_values, width, label="Graph (Neo4j)",
                    color=[RED, GREEN, BLUE], zorder=3)

for bars, vals in ((bars_kv, kv_values), (bars_graph, graph_values)):
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06,
                f"{v}/{TOTAL}", ha="center", va="bottom", fontsize=13, fontweight="bold")

ax.set_title("Blast Radius of One Poisoned Memory: Key-Value vs Graph",
             fontsize=15, fontweight="bold", pad=16)
ax.set_ylabel("Contaminated answers (of 4)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(states)
ax.set_ylim(0, 5)
ax.set_yticks(range(0, 5))
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

ax.annotate("one fact poisons\nevery traversal", xy=(0 + width / 2, 4), xytext=(0.9, 3.4),
            fontsize=11, fontweight="bold", color=RED, ha="center",
            arrowprops=dict(arrowstyle="->", color=RED, lw=2))
ax.annotate("write-gate blocks\nboth stores", xy=(1 + width / 2, 0.05), xytext=(1.4, 1.6),
            fontsize=10, fontweight="bold", color=GREEN, ha="center",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))

ax.text(0.5, -0.16,
        "Same attack (one injected boutique-hotel fact); contamination checked deterministically "
        "against the store.",
        transform=ax.transAxes, ha="center", fontsize=9, color="#666666")

plt.tight_layout()
plt.savefig("memory-hygiene-blast-radius.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: memory-hygiene-blast-radius.png")
