"""
Architecture diagram for the reasoning memory demo.

Shows how HookProvider automatically captures decision traces without
changing any tools, and why the graph store answers the reverse audit
that the flat store cannot.

Run: uv run python generate_architecture.py
Output: ai-agent-reasoning-memory-architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

TEAL   = "#5FBFA4"
CORAL  = "#F28B82"
BLUE   = "#4A90D9"
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

def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.8, label="", lpad=0.06):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=14), zorder=3)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + lpad, label, ha="center", va="bottom", fontsize=8.5,
                color=color, fontstyle="italic", zorder=5)

fig, ax = plt.subplots(figsize=(15, 7.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 15)
ax.set_ylim(0, 7.5)
ax.axis("off")

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(7.5, 7.15, "Reasoning Memory: Remember WHY the Agent Decided",
        ha="center", fontsize=16, fontweight="bold", color=NAVY)
ax.text(7.5, 6.8, "HookProvider captures decision traces automatically, zero changes to tools",
        ha="center", fontsize=10.5, color=GRAY)

# ── Agent + tools ─────────────────────────────────────────────────────────────
agent_bg = FancyBboxPatch((0.3, 3.8), 3.5, 2.5,
                           boxstyle="round,pad=0.03", linewidth=1.5,
                           facecolor="#EBF7F4", edgecolor=TEAL, zorder=3)
ax.add_patch(agent_bg)
ax.text(2.05, 6.18, "Strands Agent", ha="center", fontsize=12,
        fontweight="bold", color=TEAL, zorder=5)

tools = [("search_flights", 2.05, 5.7, SLATE),
         ("check_fare_alert", 2.05, 5.15, SLATE)]
for lbl, tx, ty, col in tools:
    rbox(ax, tx, ty, 2.7, 0.44, lbl, color=col, tc=WHITE, fs=9.5)

ax.text(2.05, 4.78, "hooks=[DecisionTraceRecorder()]", ha="center",
        fontsize=8.5, color=TEAL, fontstyle="italic", zorder=5)
hook_box = FancyBboxPatch((0.55, 4.42), 3.0, 0.55,
                           boxstyle="round,pad=0.02", linewidth=1.5,
                           facecolor=TEAL, edgecolor=WHITE, zorder=4)
ax.add_patch(hook_box)
ax.text(2.05, 4.7, "HookProvider", ha="center", fontsize=10,
        fontweight="bold", color=WHITE, zorder=5)
ax.text(2.05, 4.49, "BeforeInvocation · AfterToolCall · AfterInvocation",
        ha="center", fontsize=7.5, color=WHITE, zorder=5)

# User question
rbox(ax, 1.6, 3.35, 2.2, 0.65, "User question", "5 travel decisions", NAVY, WHITE, 10)
arrow(ax, 1.6, 3.67, 2.05, 3.8, NAVY)

# ── Decision Trace ────────────────────────────────────────────────────────────
arrow(ax, 3.8, 5.05, 5.2, 5.05, TEAL, label="records automatically")

trace_bg = FancyBboxPatch((5.2, 3.6), 3.2, 2.8,
                           boxstyle="round,pad=0.03", linewidth=1.5,
                           facecolor="#F0FFF4", edgecolor=TEAL, zorder=3)
ax.add_patch(trace_bg)
ax.text(6.8, 6.26, "Decision Trace", ha="center", fontsize=12,
        fontweight="bold", color=TEAL, zorder=5)

trace_rows = [
    ("question:", '"Should I fly JFK→MAD?"', SLATE, SLATE),
    ("steps:", "search_flights → evidence", SLATE, SLATE),
    ("evidence:", "fare_alerts_feed ← source", ORANGE, ORANGE),
    ("outcome:", '"Iberia at $892 (fare alert)"', TEAL, TEAL),
]
for i, (key, val, kc, vc) in enumerate(trace_rows):
    y = 5.85 - i * 0.48
    ax.text(5.6, y, key, ha="left", fontsize=8.5, color=GRAY, fontweight="bold", zorder=5)
    ax.text(7.1, y, val, ha="left", fontsize=8.5, color=vc, fontstyle="italic", zorder=5)

ax.text(6.8, 3.75, "One trace per invocation → agent.state", ha="center",
        fontsize=8, color=GRAY, zorder=5, fontstyle="italic")

# ── Two stores ────────────────────────────────────────────────────────────────
arrow(ax, 8.4, 5.4, 9.7, 5.7, SLATE, label="replay")
arrow(ax, 8.4, 4.6, 9.7, 3.9, TEAL, label="as graph edges")

# KV store
rbox(ax, 10.9, 5.7, 2.4, 0.7, "Key-value store", "flat trace blobs", SLATE, WHITE, 10, 8.5)

# Graph store
rbox(ax, 10.9, 3.9, 2.4, 0.7, "Neo4j graph store",
     "(:Decision)→(:Step)→(:Evidence)→(:Source)", TEAL, WHITE, 10, 7.5)

# ── Audit results ─────────────────────────────────────────────────────────────
# "Why did I decide X?"
rbox(ax, 13.5, 5.7, 2.6, 0.6, '"Why did I decide X?"', "direct replay", SLATE, WHITE, 9.5, 8)
arrow(ax, 12.1, 5.7, 12.22, 5.7, SLATE)
ax.text(13.5, 5.28, "✓ both stores answer", ha="center",
        fontsize=9, color=SLATE, fontweight="bold", zorder=5)

# Reverse audit
rev_bg = FancyBboxPatch((12.22, 2.8), 2.6, 0.9,
                         boxstyle="round,pad=0.02", linewidth=1.5,
                         facecolor="#FFF8EE", edgecolor=ORANGE, zorder=4)
ax.add_patch(rev_bg)
ax.text(13.52, 3.52, "Reverse audit", ha="center", fontsize=10,
        fontweight="bold", color=ORANGE, zorder=5)
ax.text(13.52, 3.2, '"fare_alerts_feed was wrong -', ha="center",
        fontsize=8.5, color=SLATE, zorder=5)
ax.text(13.52, 2.95, 'what decisions relied on it?"', ha="center",
        fontsize=8.5, color=SLATE, zorder=5)

arrow(ax, 12.1, 3.9, 12.22, 3.5, ORANGE)

# Results
ax.text(13.52, 2.45, "Flat scan:  2/4 found (direct only)",
        ha="center", fontsize=9.5, color=CORAL, fontweight="bold", zorder=5)
ax.text(13.52, 2.12, "DERIVED_FROM traversal: 4/4 ✓",
        ha="center", fontsize=9.5, color=TEAL, fontweight="bold", zorder=5)
ax.text(13.52, 1.75, "Graph follows evidence through\nother decisions' outputs",
        ha="center", fontsize=8.5, color=GRAY, fontstyle="italic", zorder=5)

plt.tight_layout(pad=0.3)
plt.savefig("ai-agent-reasoning-memory-architecture.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
print("Saved: ai-agent-reasoning-memory-architecture.png")
