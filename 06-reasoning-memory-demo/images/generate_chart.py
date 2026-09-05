"""
Generate the reverse-audit chart for the reasoning memory demo.

Numbers come from test_reasoning_memory.py (measured on the live demo): of 4 decisions
whose evidence chain reaches the compromised source, how many each store's audit finds.
Ground truth is fixed by construction of the seeded traces, deterministic checks, no
LLM judge, no invented numbers.

Run: uv run python generate_chart.py
Output: reasoning-memory-reverse-audit.png
"""

import matplotlib.pyplot as plt
import numpy as np

RED = "#F28B82"      # missed dependencies
GREEN = "#5FBFA4"    # found dependencies
GREY = "#B0B7C3"

TOTAL = 4  # decisions whose evidence chain reaches the compromised source

stores = ["Key-value\n(flat scan)", "Graph\n(provenance traversal)"]
found = [2, 4]     # from test_reasoning_memory.py
missed = [2, 0]

x = np.arange(len(stores))
width = 0.5

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars_found = ax.bar(x, found, width, label="Affected decisions found", color=GREEN, zorder=3)
bars_missed = ax.bar(x, missed, width, bottom=found, label="Missed (indirect dependencies)",
                     color=RED, zorder=3)

for i, (f, m) in enumerate(zip(found, missed)):
    ax.text(i, f / 2, f"{f}/{TOTAL}", ha="center", va="center",
            fontsize=15, fontweight="bold", color="white")
    if m:
        ax.text(i, f + m / 2, f"missed {m}", ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")

ax.set_title('Reverse Audit: "This Source Was Wrong, Which Decisions Depended on It?"',
             fontsize=14, fontweight="bold", pad=16)
ax.set_ylabel(f"Affected decisions (of {TOTAL})", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(stores, fontsize=12)
ax.set_ylim(0, 5)
ax.set_yticks(range(0, 5))
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

ax.annotate("a flat blob never mentions\nsources it depends on indirectly",
            xy=(0, 3.6), xytext=(0.52, 4.4),
            fontsize=10, fontweight="bold", color=RED, ha="center",
            arrowprops=dict(arrowstyle="->", color=RED, lw=2))
ax.annotate("DERIVED_FROM*0.. follows\nprovenance at any depth",
            xy=(1, 4.0), xytext=(1.25, 4.55),
            fontsize=10, fontweight="bold", color=GREEN, ha="center",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))

ax.text(0.5, -0.16,
        "Same recorded decision traces in both stores; 2 of the 4 affected decisions depend on the "
        "source only through other decisions' outputs.",
        transform=ax.transAxes, ha="center", fontsize=9, color="#666666")

plt.tight_layout()
plt.savefig("reasoning-memory-reverse-audit.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: reasoning-memory-reverse-audit.png")
