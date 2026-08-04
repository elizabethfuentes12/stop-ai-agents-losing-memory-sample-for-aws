"""
Generate the multi-hop recall chart for the graph memory demo.

Numbers come from the deterministic scorecard in test_graph_memory.py (Test 4):
4 multi-hop questions over the same known graph, checked against the known answer.
Reproducible — no LLM judge, no invented benchmark numbers.

Run: uv run python generate_chart.py
Output: graph-memory-multihop.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RED = "#F28B82"
GREEN = "#5FBFA4"

# From test_graph_memory.py Test 4 (deterministic scorecard), measured on the seeded graph.
TOTAL = 4
BEFORE_HITS = 1   # semantic recall (pure vector similarity)
AFTER_HITS = 4    # graph recall (vector similarity + traversal)

strategies = [
    "Semantic recall\n(before)\nvector similarity",
    "Graph recall\n(after)\nsimilarity + traversal",
]
values = [BEFORE_HITS, AFTER_HITS]
colors = [RED, GREEN]

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars = ax.bar(strategies, values, color=colors, width=0.5, zorder=3)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.08,
        f"{value}/{TOTAL}",
        ha="center", va="bottom",
        fontsize=16, fontweight="bold",
    )

ax.set_title(
    "Multi-hop Questions Answered Correctly: Semantic vs Graph Memory",
    fontsize=15, fontweight="bold", pad=16,
)
ax.set_ylabel("Correct answers (of 4 multi-hop questions)", fontsize=12)
ax.set_ylim(0, 5)
ax.set_yticks(range(0, 5))
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

legend_patches = [
    mpatches.Patch(color=RED, label="Semantic — finds related pieces, can't connect them"),
    mpatches.Patch(color=GREEN, label="Graph — traverses relationships to the answer"),
]
ax.legend(handles=legend_patches, loc="upper left", fontsize=10, framealpha=0.9)

ax.annotate(
    "traversal recovers\nthe full chain",
    xy=(1, AFTER_HITS), xytext=(0.55, 3.4),
    fontsize=11, fontweight="bold", color=GREEN, ha="center",
    arrowprops=dict(arrowstyle="->", color=GREEN, lw=2),
)

ax.text(
    0.5, -0.16,
    'Same seeded graph (Sarah Chen → Vista Hotels → boutique → Kyoto → Japan); '
    "answer checked against the known graph.",
    transform=ax.transAxes, ha="center", fontsize=9, color="#666666",
)

plt.tight_layout()
plt.savefig("graph-memory-multihop.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: graph-memory-multihop.png")
