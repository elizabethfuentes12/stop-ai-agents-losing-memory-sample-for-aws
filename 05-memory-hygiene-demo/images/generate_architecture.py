"""
Architecture diagram for the memory hygiene demo.

Shows the write-gate blocking poisoned content before it reaches memory,
and how blast radius differs between key-value and graph backends.

Run: uv run python generate_architecture.py
Output: ai-agent-memory-hygiene-architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

TEAL   = "#5FBFA4"
RED    = "#F28B82"
GREEN  = "#48BB78"
SLATE  = "#4A5568"
NAVY   = "#2D3748"
ORANGE = "#F5A623"
GRAY   = "#718096"
WHITE  = "#FFFFFF"
BG     = "#F7F9FC"
PURPLE = "#7B4F9E"

def rbox(ax, cx, cy, w, h, label, sub="", color=TEAL, tc=WHITE, fs=10.5, subfs=8.5):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.02", linewidth=1.5,
                          facecolor=color, edgecolor=WHITE, zorder=4)
    ax.add_patch(box)
    ty = cy + h*0.14 if sub else cy
    ax.text(cx, ty, label, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=tc, zorder=5)
    if sub:
        ax.text(cx, cy - h*0.18, sub, ha="center", va="center", fontsize=subfs,
                color=tc, alpha=0.9, zorder=5)

def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.8, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=14), zorder=3)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.06, label, ha="center", va="bottom", fontsize=8.5,
                color=color, fontstyle="italic", zorder=5)

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(7.0, 6.65, "Memory Hygiene: Write-Gate Stops Poison at the Source",
        ha="center", fontsize=16, fontweight="bold", color=NAVY)
ax.text(7.0, 6.3, "Screen at write time, then forget what already got in",
        ha="center", fontsize=10.5, color=GRAY)

# ── Inputs ────────────────────────────────────────────────────────────────────
rbox(ax, 1.4, 5.3, 2.0, 0.65, "Legitimate user", "trust = 1.0", GREEN, WHITE, 10)
rbox(ax, 1.4, 3.8, 2.0, 0.65, "Attacker / injection", "trust = 0.0", RED, WHITE, 10)

# ── Write-gate ────────────────────────────────────────────────────────────────
gate_bg = FancyBboxPatch((3.9, 2.4), 2.8, 3.6,
                          boxstyle="round,pad=0.03", linewidth=2,
                          facecolor="#EBF7F4", edgecolor=TEAL, zorder=3)
ax.add_patch(gate_bg)
ax.text(5.3, 5.82, "Write-gate", ha="center", fontsize=12,
        fontweight="bold", color=TEAL, zorder=5)
ax.text(5.3, 5.5, "screen_memory(content, trust)", ha="center",
        fontsize=8.5, color=SLATE, fontstyle="italic", zorder=5)

checks = [
    "✗ instruction override",
    "✗ role rewrite",
    "✗ PII shapes",
    "✗ trust < min_trust",
]
for i, chk in enumerate(checks):
    ax.text(5.3, 5.12 - i*0.37, chk, ha="center", fontsize=8.5,
            color=SLATE, zorder=5)

ax.plot([4.2, 6.4], [2.75, 2.75], color=TEAL, lw=1, alpha=0.4, linestyle="--", zorder=4)
ax.text(5.3, 2.6, "allowed = True / False", ha="center", fontsize=8,
        color=TEAL, fontstyle="italic", zorder=5)

# Arrows into gate
arrow(ax, 2.42, 5.3, 3.9, 5.0, GREEN, label="stores fact")
arrow(ax, 2.42, 3.8, 3.9, 4.0, RED, label="tries to inject")

# ── BLOCKED path ──────────────────────────────────────────────────────────────
arrow(ax, 5.3, 2.4, 5.3, 1.65, RED)
blocked = FancyBboxPatch((4.0, 0.95), 2.6, 0.65,
                          boxstyle="round,pad=0.02", linewidth=1.5,
                          facecolor="#FEE2E2", edgecolor=RED, zorder=4)
ax.add_patch(blocked)
ax.text(5.3, 1.27, "✗  BLOCKED, not written to memory", ha="center",
        fontsize=9.5, color=RED, fontweight="bold", zorder=5)

# ── ALLOWED path ──────────────────────────────────────────────────────────────
arrow(ax, 6.7, 4.4, 8.0, 4.8, GREEN, label="allowed")

# ── KV Store ──────────────────────────────────────────────────────────────────
rbox(ax, 9.2, 5.15, 2.3, 0.7, "Key-value store", "agent.state", SLATE, WHITE, 10, 8.5)
arrow(ax, 8.0, 4.8, 8.1, 5.15, GREEN)

blast_kv = FancyBboxPatch((8.05, 4.0), 2.3, 0.72,
                           boxstyle="round,pad=0.02", linewidth=1.5,
                           facecolor="#FAFAFA", edgecolor=SLATE, zorder=4)
ax.add_patch(blast_kv)
ax.text(9.2, 4.52, "Blast radius if poisoned:", ha="center",
        fontsize=8.5, color=SLATE, zorder=5)
ax.text(9.2, 4.18, "1/4, one blob, one key", ha="center",
        fontsize=9, color=SLATE, fontweight="bold", zorder=5)

# ── Graph Store ───────────────────────────────────────────────────────────────
rbox(ax, 9.2, 3.2, 2.3, 0.7, "Neo4j graph store", "nodes + edges", TEAL, WHITE, 10, 8.5)
arrow(ax, 8.0, 4.8, 8.1, 3.2, GREEN)

blast_graph = FancyBboxPatch((8.05, 2.05), 2.3, 0.72,
                              boxstyle="round,pad=0.02", linewidth=1.5,
                              facecolor="#FAFAFA", edgecolor=TEAL, zorder=4)
ax.add_patch(blast_graph)
ax.text(9.2, 2.57, "Blast radius if poisoned:", ha="center",
        fontsize=8.5, color=SLATE, zorder=5)
ax.text(9.2, 2.23, "4/4, propagates via edges", ha="center",
        fontsize=9, color=TEAL, fontweight="bold", zorder=5)

# ── Forget path ───────────────────────────────────────────────────────────────
rbox(ax, 12.0, 4.15, 2.2, 0.65, "Forget", "del or DETACH DELETE", ORANGE, WHITE, 10, 8.5)
arrow(ax, 10.35, 4.87, 11.0, 4.3, ORANGE, label="cleanup")
arrow(ax, 10.35, 3.2, 11.0, 4.0, ORANGE)

clean = FancyBboxPatch((10.9, 1.15), 2.2, 0.65,
                        boxstyle="round,pad=0.02", linewidth=1.5,
                        facecolor="#FFF8EE", edgecolor=ORANGE, zorder=4)
ax.add_patch(clean)
ax.text(12.0, 1.47, "After forget: 0/4 in both stores", ha="center",
        fontsize=9.5, color=ORANGE, fontweight="bold", zorder=5)
arrow(ax, 12.0, 3.82, 12.0, 1.82, ORANGE)

plt.tight_layout(pad=0.3)
plt.savefig("ai-agent-memory-hygiene-architecture.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
print("Saved: ai-agent-memory-hygiene-architecture.png")
