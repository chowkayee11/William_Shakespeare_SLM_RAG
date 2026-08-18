#!/usr/bin/env python3
"""
Generate a conference-style architecture diagram.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse
import matplotlib as mpl

# Set style - conference-like
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(14, 9), dpi=200)
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

# Colors - conference style (professional blues/grays)
COLORS = {
    'primary': '#336699',
    'primary_light': '#e6f0fa',
    'secondary': '#666666',
    'secondary_light': '#f5f5f5',
    'accent': '#339966',
    'accent_light': '#e6f5ec',
    'arrow': '#555555'
}

# Helper function to draw process boxes
def draw_process(x, y, w, h, text, color=COLORS['primary_light']):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="square,pad=0",
                         facecolor=color,
                         edgecolor=COLORS['primary'],
                         linewidth=1.5,
                         zorder=3)
    ax.add_patch(box)
    lines = text.split('\\n')
    line_h = 0.22
    start_y = y + h/2 + (len(lines)-1)*line_h/2
    for i, line in enumerate(lines):
        ax.text(x + w/2, start_y - i*line_h, line,
                ha='center', va='center', fontsize=9,
                color='#222222')

# Helper function to draw storage boxes
def draw_storage(x, y, w, h, text):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="square,pad=0",
                         facecolor=COLORS['secondary_light'],
                         edgecolor=COLORS['secondary'],
                         linewidth=1.5,
                         zorder=3)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center', fontsize=9,
            color='#222222')

# Helper function to draw I/O ellipses
def draw_io(x, y, w, h, text, color=COLORS['accent_light']):
    ellipse = Ellipse((x + w/2, y + h/2), w, h,
                      facecolor=color,
                      edgecolor=COLORS['accent'],
                      linewidth=1.5,
                      zorder=3)
    ax.add_patch(ellipse)
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center', fontsize=9,
            color='#222222')

# Helper function to draw arrows
def draw_arrow(x1, y1, x2, y2, linestyle='-', color=COLORS['arrow'], linewidth=1.5):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->,head_width=0.15,head_length=0.2',
                           color=color,
                           linestyle=linestyle,
                           linewidth=linewidth,
                           zorder=2)
    ax.add_patch(arrow)

# Helper function to draw contribution bounding boxes
def draw_contrib_box(x, y, w, h, label):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.05",
                         facecolor='none',
                         edgecolor=COLORS['primary'],
                         linewidth=2,
                         linestyle='--',
                         zorder=4)
    ax.add_patch(box)
    ax.text(x + w - 0.05, y + h - 0.05, label,
            ha='right', va='top', fontsize=8,
            fontweight='bold', color=COLORS['primary'])

# ============================================================================
# Offline Pipeline (Left)
# ============================================================================

# Labels
ax.text(2.5, 8.6, 'Offline Indexing', ha='center',
        fontsize=11, fontweight='bold', color=COLORS['primary'])

# 1. Raw Dataset
draw_io(1.0, 7.5, 1.8, 0.6, 'Raw Dataset')

# 2. Data Loading
draw_process(1.0, 6.5, 1.8, 0.7, 'Data Loading')

# 3. Scene-level Chunking
draw_process(1.0, 5.5, 1.8, 0.7, 'Scene-level\\nChunking')

# 4. Metadata Enrichment
draw_process(1.0, 4.5, 1.8, 0.7, 'Metadata\\nEnrichment')

# 5. Hybrid Embedding
draw_process(1.0, 3.5, 1.8, 0.7, 'Hybrid\\nEmbedding')

# 6. Vector Index
draw_storage(1.0, 2.4, 1.8, 0.7, 'Vector Index')

# ============================================================================
# Online Pipeline (Right)
# ============================================================================

ax.text(9.5, 8.6, 'Online Inference', ha='center',
        fontsize=11, fontweight='bold', color=COLORS['primary'])

# 1. User Query
draw_io(8.5, 7.5, 1.8, 0.6, 'User Query')

# 2. Top-k Retrieval
draw_process(8.5, 6.3, 1.8, 0.7, 'Top-$k$\\nRetrieval')

# 3. Prompt Assembly
draw_process(8.5, 5.1, 1.8, 0.7, 'Prompt\\nAssembly')

# 4. SLM Generation
draw_process(8.5, 3.9, 1.8, 0.7, 'SLM\\nGeneration')

# 5. Grounded Answer
draw_io(8.5, 2.7, 1.8, 0.6, 'Grounded Answer')

# ============================================================================
# Arrows
# ============================================================================

# Offline arrows
draw_arrow(1.9, 8.0, 1.9, 7.2)
draw_arrow(1.9, 7.1, 1.9, 6.2)
draw_arrow(1.9, 6.1, 1.9, 5.2)
draw_arrow(1.9, 5.1, 1.9, 4.2)
draw_arrow(1.9, 4.1, 1.9, 3.1)

# Online arrows
draw_arrow(9.4, 8.0, 9.4, 7.0)
draw_arrow(9.4, 6.2, 9.4, 5.8)
draw_arrow(9.4, 5.0, 9.4, 4.6)
draw_arrow(9.4, 3.8, 9.4, 3.3)

# Index to retrieval
draw_arrow(2.8, 2.75, 5.0, 2.75)
draw_arrow(5.0, 2.75, 5.0, 6.65)
draw_arrow(5.0, 6.65, 8.5, 6.65)

# Baseline (bypass)
draw_arrow(10.3, 7.8, 12.0, 7.8, linestyle=':', color=COLORS['primary'])
draw_arrow(12.0, 7.8, 12.0, 4.25, linestyle=':', color=COLORS['primary'])
draw_arrow(12.0, 4.25, 10.3, 4.25, linestyle=':', color=COLORS['primary'])
ax.text(12.1, 6.0, 'Baseline', ha='left', va='center',
        fontsize=8, color=COLORS['primary'])

# ============================================================================
# Contribution Boxes
# ============================================================================

# Contribution 1: Chunking + Enrichment
draw_contrib_box(0.8, 4.3, 2.2, 2.2, 'Contribution 1')

# Contribution 2: Hybrid Embedding
draw_contrib_box(0.8, 3.3, 2.2, 1.0, 'Contribution 2')

# Contribution 3: SLM Generation
draw_contrib_box(8.3, 3.7, 2.2, 1.0, 'Contribution 3')

# Title
ax.text(7, 9.0, 'Shakespeare SLM/RAG System Architecture', ha='center',
        fontsize=13, fontweight='bold', color='#222222')

plt.tight_layout()
plt.savefig('c:\\Users\\ZhuanZ\\William_Shakespeare_SLM_RAG\\report\\architecture_diagram_v2.png',
            dpi=250, bbox_inches='tight', facecolor='white')
print("Conference-style diagram saved to: report/architecture_diagram_v2.png")

plt.show()
