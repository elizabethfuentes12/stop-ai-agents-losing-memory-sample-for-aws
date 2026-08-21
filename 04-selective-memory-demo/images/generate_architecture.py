"""
Architecture diagram for the selective memory demo.

Shows the 3 selection mechanisms side by side:
  A — agent tools inline (couples selection to turn latency)
  B — own extractor off-path (4 typed prompts → S3 Vectors)
  C — AgentCore managed (async, ~53-85 s lag)

Run: uv run python generate_architecture.py
Output: ai-agent-selective-memory-architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

TEAL   = "#5FBFA4"
BLUE   = "#4A90D9"
ORANGE = "#F5A623"
SLATE  = "#4A5568"
NAVY   = "#2D3748"
GRAY   = "#718096"
WHITE  = "#FFFFFF"
BG     = "#F7F9FC"
GREEN  = "#38A169"
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
                color=tc, alpha=0.88, zorder=5)

def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.8, label="", dashed=False):
    ls = "--" if dashed else "-"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, mutation_scale=13), zorder=3)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.04, label, ha="center", va="bottom", fontsize=8,
                color=color, fontstyle="italic", zorder=5)

fig, ax = plt.subplots(figsize=(15, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 15)
ax.set_ylim(0, 8)
ax.axis("off")

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(7.5, 7.65, "3 Ways to Select What an AI Agent Remembers", ha="center",
        fontsize=17, fontweight="bold", color=NAVY)
ax.text(7.5, 7.25, "Same conversation, same ground truth (5 keepers + 3 decoys) — only the selection mechanism changes",
        ha="center", fontsize=10.5, color=GRAY)

# ── Input: conversation ───────────────────────────────────────────────────────
rbox(ax, 1.4, 4.4, 2.3, 3.2,
     "Conversation", "6 turns · 5 keepers\n3 decoys planted",
     SLATE, WHITE, 11, 9)

# ── Column headers ────────────────────────────────────────────────────────────
cols = [
    (4.7,  "A — Inline", "Agent tools",       TEAL),
    (8.0,  "B — Off-path", "Own extractor",    BLUE),
    (11.5, "C — Managed", "AgentCore",         ORANGE),
]
for cx, h1, h2, col in cols:
    ax.text(cx, 7.0, h1, ha="center", fontsize=13, fontweight="bold", color=col)
    ax.text(cx, 6.72, h2, ha="center", fontsize=10, color=col)
    ax.plot([cx - 1.3, cx + 1.3], [6.6, 6.6], color=col, lw=1.5, alpha=0.4)

# ── Mechanism A ───────────────────────────────────────────────────────────────
arrow(ax, 2.56, 5.5, 3.7, 5.5, TEAL, label="every turn")

rbox(ax, 4.7, 5.5, 2.3, 0.75, "Agent selects inline",
     "core_memory_write / read", TEAL, WHITE)

arrow(ax, 4.7, 5.12, 4.7, 4.5, TEAL)
rbox(ax, 4.7, 4.15, 2.0, 0.55, "agent.state", "key-value store", TEAL, WHITE, 10, 8)

arrow(ax, 4.7, 3.87, 4.7, 3.4, TEAL)
# Results A
res_a = FancyBboxPatch((3.55, 2.4), 2.3, 0.88,
                        boxstyle="round,pad=0.02", linewidth=1.5,
                        facecolor="#EBF7F4", edgecolor=TEAL, zorder=4)
ax.add_patch(res_a)
ax.text(4.7, 3.18, "4/5 kept · 0 decoys", ha="center", fontsize=10,
        color=TEAL, fontweight="bold", zorder=5)
ax.text(4.7, 2.74, "~1.5-2.5 s/turn overhead\nAvailable: immediately", ha="center",
        fontsize=8.5, color=SLATE, zorder=5)

# ── Mechanism B ───────────────────────────────────────────────────────────────
arrow(ax, 2.56, 4.4, 7.0, 4.4, BLUE, label="off-path (no turn delay)")

rbox(ax, 8.0, 4.4, 2.5, 0.75, "4 extraction prompts",
     "facts · prefs · summary · episodes", BLUE, WHITE, 9.5, 8)

arrow(ax, 8.0, 4.02, 8.0, 3.4, BLUE)
rbox(ax, 8.0, 3.1, 2.2, 0.55, "Amazon S3 Vectors",
     "4 typed indexes · Titan V2", BLUE, WHITE, 10, 8)

arrow(ax, 8.0, 2.82, 8.0, 2.3, BLUE)
res_b = FancyBboxPatch((6.85, 1.3), 2.3, 0.88,
                        boxstyle="round,pad=0.02", linewidth=1.5,
                        facecolor="#EBF4FF", edgecolor=BLUE, zorder=4)
ax.add_patch(res_b)
ax.text(8.0, 2.08, "5/5 kept · 0 decoys  [best quality]", ha="center", fontsize=10,
        color=BLUE, fontweight="bold", zorder=5)
ax.text(8.0, 1.64, "0 s overhead · off-path\nAvailable: ~4 s/turn", ha="center",
        fontsize=8.5, color=SLATE, zorder=5)

# ── Mechanism C ───────────────────────────────────────────────────────────────
arrow(ax, 2.56, 3.3, 10.4, 3.3, ORANGE, label="create_event (async)")

rbox(ax, 11.5, 3.3, 2.5, 0.75, "AgentCore Memory",
     "4 built-in strategies · AWS managed", ORANGE, WHITE, 9.5, 8)

arrow(ax, 11.5, 2.92, 11.5, 2.3, ORANGE)
res_c = FancyBboxPatch((10.35, 1.3), 2.3, 0.88,
                        boxstyle="round,pad=0.02", linewidth=1.5,
                        facecolor="#FFF8EE", edgecolor=ORANGE, zorder=4)
ax.add_patch(res_c)
ax.text(11.5, 2.08, "5/5 kept · 0-2 decoys*", ha="center", fontsize=10,
        color=ORANGE, fontweight="bold", zorder=5)
ax.text(11.5, 1.64, "~0.4 s/turn overhead\nAvailable: ~53-85 s  (measured)", ha="center",
        fontsize=8.5, color=SLATE, zorder=5)

# ── Footnote ──────────────────────────────────────────────────────────────────
ax.text(7.5, 0.9, "* C is nondeterministic — extraction criteria are managed by AWS, not user-controlled",
        ha="center", fontsize=8.5, color=GRAY, fontstyle="italic")
ax.text(7.5, 0.6, "B is 'AgentCore built by hand': same pipeline, same types — you own the extraction prompts",
        ha="center", fontsize=8.5, color=GRAY, fontstyle="italic")

plt.tight_layout(pad=0.3)
plt.savefig("ai-agent-selective-memory-architecture.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
print("Saved: ai-agent-selective-memory-architecture.png")
