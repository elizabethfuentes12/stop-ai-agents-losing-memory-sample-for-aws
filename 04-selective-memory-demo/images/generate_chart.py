"""
Generate the selection-quality chart for the selective-memory demo.

Numbers come from test_selective_memory.py (measured on the live demo):
items kept, decoys leaked, and availability lag per mechanism.
Deterministic scoring against planted ground truth — no LLM judge.

Run: uv run python generate_chart.py
Output: selective-memory-mechanisms.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

TEAL   = "#5FBFA4"
ORANGE = "#F5A623"
BLUE   = "#7EA6E0"
GRAY   = "#B0B7C3"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.patch.set_facecolor("white")

# ── Left panel: selection quality (kept / decoys) ────────────────────────────
mechanisms = ["A — agent\ntools", "B — own\nextractor", "C — AgentCore\nmanaged"]
kept   = [4, 5, 5]     # items correctly kept (of 5)
decoys = [0, 0, 1]     # decoys that leaked (C is nondeterministic; median observed)
TOTAL  = 5

x = np.arange(len(mechanisms))
w = 0.34

for ax in (ax1, ax2):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

bars_kept  = ax1.bar(x - w / 2, kept,   w, label="Kept (of 5)",      color=[TEAL, TEAL, TEAL], zorder=3)
bars_decoy = ax1.bar(x + w / 2, decoys, w, label="Decoys leaked",    color=[GRAY, GRAY, ORANGE], zorder=3)

for bar, v in zip(bars_kept, kept):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
             f"{v}/5", ha="center", va="bottom", fontsize=13, fontweight="bold")
for bar, v in zip(bars_decoy, decoys):
    label = f"{v}" if v == 0 else f"{v}*"
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
             label, ha="center", va="bottom", fontsize=13, fontweight="bold",
             color=ORANGE if v > 0 else "black")

ax1.set_title("Selection Quality", fontsize=14, fontweight="bold", pad=14)
ax1.set_ylabel("Items (of 5)", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(mechanisms, fontsize=10)
ax1.set_ylim(0, 6.5)
ax1.set_yticks(range(0, 6))
ax1.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax1.text(0.5, -0.18, "* C is nondeterministic — 0-2 decoys across runs",
         transform=ax1.transAxes, ha="center", fontsize=9, color="#666666")

# ── Right panel: availability lag ────────────────────────────────────────────
# A: 0 s (immediately in-turn), B: ~4 s (off-path extractor), C: ~69 s (median measured)
lags = [0, 4, 69]
colors = [TEAL, BLUE, ORANGE]

bars_lag = ax2.bar(x, lags, 0.5, color=colors, zorder=3)
labels = ["0 s\n(immediate)", "~4 s\n(off-path)", "~53-85 s\n(managed)"]
for bar, v, lbl in zip(bars_lag, lags, labels):
    ypos = bar.get_height() + 1.5
    ax2.text(bar.get_x() + bar.get_width() / 2, ypos,
             lbl, ha="center", va="bottom", fontsize=11, fontweight="bold")

ax2.set_title("Memory Availability After Conversation Turn", fontsize=14, fontweight="bold", pad=14)
ax2.set_ylabel("Seconds until memory is queryable", fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(mechanisms, fontsize=10)
ax2.set_ylim(0, 100)
ax2.text(0.5, -0.18, "A couples latency to the conversation; B and C are off-path.",
         transform=ax2.transAxes, ha="center", fontsize=9, color="#666666")

# ── Patches for the right panel legend ───────────────────────────────────────
patches = [mpatches.Patch(color=c, label=l)
           for c, l in [(TEAL, "A — inline"), (BLUE, "B — extractor"), (ORANGE, "C — AgentCore")]]
ax2.legend(handles=patches, loc="upper left", fontsize=10, framealpha=0.9)

fig.suptitle("3 Ways to Select What an AI Agent Remembers",
             fontsize=16, fontweight="bold", y=1.02)

plt.tight_layout()
plt.savefig("selective-memory-mechanisms.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: selective-memory-mechanisms.png")
