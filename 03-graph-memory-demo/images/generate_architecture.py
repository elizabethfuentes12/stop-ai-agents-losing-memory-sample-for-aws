"""
Architecture diagram for the graph memory demo.

Shows the two retrieval paths an agent can take with the same Neo4j graph:
  - recall_semantic: finds related nodes by similarity — pieces, no connections
  - recall_graph: similarity to entry node → Cypher traversal → full chain back to person

Run: uv run python generate_architecture.py
Output: ai-agent-graph-memory-architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

TEAL   = "#5FBFA4"
CORAL  = "#F28B82"
SLATE  = "#4A5568"
NAVY   = "#2D3748"
ORANGE = "#F5A623"
PURPLE = "#7B4F9E"
GRAY   = "#718096"
WHITE  = "#FFFFFF"
BG     = "#F7F9FC"

def rbox(ax, cx, cy, w, h, label, sub="", color=TEAL, tc=WHITE, fs=11, subfs=9.5):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.02", linewidth=1.5,
                          facecolor=color, edgecolor=WHITE, zorder=4)
    ax.add_patch(box)
    ty = cy + h*0.12 if sub else cy
    ax.text(cx, ty, label, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=tc, zorder=5)
    if sub:
        ax.text(cx, cy - h*0.18, sub, ha="center", va="center", fontsize=subfs,
                color=tc, alpha=0.88, zorder=5)

def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.8, label="", lfs=8.5, lpad=0.025):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=14), zorder=3)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2 + lpad
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=lfs,
                color=color, fontstyle="italic", zorder=5)

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(7, 6.65, "Graph Memory for AI Agents", ha="center", va="center",
        fontsize=17, fontweight="bold", color=NAVY)
ax.text(7, 6.3, "Same graph, two retrieval strategies — only traversal connects the pieces",
        ha="center", va="center", fontsize=11, color=GRAY)

# ── User Question ─────────────────────────────────────────────────────────────
rbox(ax, 1.4, 4.4, 2.2, 0.8, "User question",
     '"Who do I know connected\nto flights to Spain?"', SLATE, WHITE, 10, 8)

# ── Agent ─────────────────────────────────────────────────────────────────────
rbox(ax, 4.3, 4.4, 2.0, 0.8, "Strands Agent", "gpt-4o-mini", NAVY, WHITE, 11)

arrow(ax, 2.52, 4.4, 3.3, 4.4, SLATE, label="asks")

# ── Two paths down ────────────────────────────────────────────────────────────
# Path A: recall_semantic (top row)
arrow(ax, 4.3, 4.0, 4.3, 3.15, CORAL, label="Path A")
rbox(ax, 4.3, 2.8, 2.1, 0.65, "recall_semantic",
     "vector similarity only", CORAL, WHITE, 10, 8.5)

# Path B: recall_graph (below)
arrow(ax, 4.3, 4.0, 7.2, 3.15, TEAL, label="Path B")
rbox(ax, 7.2, 2.8, 2.1, 0.65, "recall_graph",
     "similarity + traversal", TEAL, WHITE, 10, 8.5)

# ── Semantic result ───────────────────────────────────────────────────────────
arrow(ax, 4.3, 2.48, 4.3, 1.75, CORAL)

# Result bubble semantic
result_a = FancyBboxPatch((2.7, 0.9), 3.2, 0.75,
                           boxstyle="round,pad=0.02", linewidth=1.5,
                           facecolor="#FFF0EE", edgecolor=CORAL, zorder=4)
ax.add_patch(result_a)
ax.text(4.3, 1.51, "Found: Iberia ·  Madrid · Spain", ha="center",
        fontsize=9.5, color=CORAL, zorder=5, fontstyle="italic")
ax.text(4.3, 1.2, "Maya Torres? ✗  (no relationship traversal)", ha="center",
        fontsize=9, color=CORAL, fontweight="bold", zorder=5)

# ── Neo4j graph ───────────────────────────────────────────────────────────────
arrow(ax, 7.2, 2.48, 7.2, 1.85, TEAL)

# Graph box
graph_bg = FancyBboxPatch((5.5, 0.3), 3.4, 1.45,
                           boxstyle="round,pad=0.03", linewidth=1.5,
                           facecolor="#EBF7F4", edgecolor=TEAL, zorder=3)
ax.add_patch(graph_bg)
ax.text(7.2, 1.6, "Neo4j Knowledge Graph", ha="center", fontsize=9.5,
        color=TEAL, fontweight="bold", zorder=5)

# Mini graph nodes
nodes = [("Maya\nTorres", 6.05, 0.82, PURPLE),
         ("Iberia", 6.9, 0.82, ORANGE),
         ("Madrid", 7.75, 0.82, SLATE),
         ("Spain", 8.55, 0.82, SLATE)]
for lbl, nx, ny, nc in nodes:
    c = plt.Circle((nx, ny), 0.24, color=nc, zorder=5)
    ax.add_patch(c)
    ax.text(nx, ny, lbl, ha="center", va="center", fontsize=7.5,
            color=WHITE, fontweight="bold", zorder=6)

for i in range(len(nodes) - 1):
    x1, y1 = nodes[i][1] + 0.24, nodes[i][2]
    x2, y2 = nodes[i+1][1] - 0.24, nodes[i+1][2]
    labels = ["WORKS_AT", "FLIES_TO", "IN_COUNTRY"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.5,
                                mutation_scale=11), zorder=4)
    mx = (x1+x2)/2
    ax.text(mx, y1 + 0.15, labels[i], ha="center", fontsize=6.5,
            color=TEAL, zorder=6)

# ── Graph result ──────────────────────────────────────────────────────────────
arrow(ax, 8.9, 1.15, 10.2, 2.0, TEAL)

result_b = FancyBboxPatch((9.2, 1.8), 3.5, 0.9,
                           boxstyle="round,pad=0.02", linewidth=1.5,
                           facecolor="#EBF7F4", edgecolor=TEAL, zorder=4)
ax.add_patch(result_b)
ax.text(10.95, 2.56, "Answer: Maya Torres", ha="center",
        fontsize=10, color=TEAL, fontweight="bold", zorder=5)
ax.text(10.95, 2.2, "Maya Torres → Iberia → Madrid → Spain", ha="center",
        fontsize=9, color=SLATE, zorder=5, fontstyle="italic")
ax.text(10.95, 1.92, "Maya Torres? ✓  (traversal followed the edges)", ha="center",
        fontsize=9, color=TEAL, fontweight="bold", zorder=5)

# ── Back to agent ─────────────────────────────────────────────────────────────
arrow(ax, 10.95, 2.72, 5.3, 4.25, TEAL, label="returns answer")

# ── Scorecard callout ─────────────────────────────────────────────────────────
score_bg = FancyBboxPatch((10.0, 3.8), 3.6, 1.4,
                           boxstyle="round,pad=0.03", linewidth=1.5,
                           facecolor=WHITE, edgecolor=NAVY, zorder=4)
ax.add_patch(score_bg)
ax.text(11.8, 5.06, "Scorecard (4 questions)", ha="center",
        fontsize=9.5, fontweight="bold", color=NAVY, zorder=5)
ax.text(11.8, 4.75, "recall_semantic:  1/4 correct", ha="center",
        fontsize=9.5, color=CORAL, zorder=5)
ax.text(11.8, 4.45, "recall_graph:      4/4 correct", ha="center",
        fontsize=9.5, color=TEAL, fontweight="bold", zorder=5)
ax.text(11.8, 4.02, "Same graph · same vector index · same agent",
        ha="center", fontsize=8.5, color=GRAY, zorder=5, fontstyle="italic")

plt.tight_layout(pad=0.3)
plt.savefig("ai-agent-graph-memory-architecture.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
print("Saved: ai-agent-graph-memory-architecture.png")
