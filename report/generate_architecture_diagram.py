#!/usr/bin/env python3
"""
Generate a simple but clear architecture diagram for the Shakespeare RAG system.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib as mpl

# Set style
plt.style.use('default')
fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
COLORS = {
    'data': '#e8f5ff',
    'chunk': '#e8f5ff',
    'retrieval': '#fff3e0',
    'prompt': '#ffe8ff',
    'llm': '#ffe8ff',
    'output': '#ffffdc',
    'arrow': '#333333',
    'highlight': '#ff4444',
    'text': '#222222'
}

# Helper function to draw boxes
def draw_box(x, y, w, h, text, color, is_highlight=False, fontsize=10):
    box = FancyBboxPatch((x, y), w, h, 
                         boxstyle="round,pad=0.05", 
                         facecolor=color, 
                         edgecolor='#333333' if not is_highlight else '#ff4444',
                         linewidth=2 if is_highlight else 1.5,
                         zorder=2)
    ax.add_patch(box)
    
    # Add text
    lines = text.split('\\n')
    line_height = 0.25
    start_y = y + h/2 + (len(lines)-1)*line_height/2
    for i, line in enumerate(lines):
        ax.text(x + w/2, start_y - i*line_height, line, 
                ha='center', va='center', fontsize=fontsize, 
                fontweight='bold' if is_highlight else 'normal',
                color=COLORS['text'])

# Helper function to draw arrows
def draw_arrow(x1, y1, x2, y2, color=COLORS['arrow'], style='->', linestyle='-', linewidth=1.5):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                           arrowstyle=style, 
                           color=color, 
                           linestyle=linestyle,
                           linewidth=linewidth,
                           mutation_scale=15,
                           zorder=1)
    ax.add_patch(arrow)

# ============================================================================
# Column 1: Offline Pipeline (left)
# ============================================================================

# Labels
ax.text(2.5, 9.3, 'Offline Indexing', ha='center', fontsize=12, fontweight='bold', color='#555555')

# 1. Data Loader
draw_box(1, 8, 3, 0.8, 'Dataset Loader\\n(data_loader.py)', COLORS['data'])

# 2. Chunking Engine (Highlight 1)
draw_box(1, 6.7, 3, 1.0, 'Chunking Engine\\n(chunking.py)', COLORS['chunk'], is_highlight=True, fontsize=9)
draw_box(0.5, 5.6, 1.8, 0.7, 'Metadata Enrichment', COLORS['chunk'], fontsize=8)

# 3. Embedding & Index
draw_box(1.7, 4.2, 2.5, 0.8, 'Dense Embedding\\n(all-MiniLM-L6-v2)', COLORS['retrieval'])
draw_box(1.7, 3.2, 2.5, 0.8, 'Sparse TF-IDF\\n(ngram 1-2)', COLORS['retrieval'])
draw_box(1, 1.8, 3, 0.9, 'Hybrid Index Store\\n(retrieval.py)', COLORS['retrieval'], is_highlight=True, fontsize=9)

# ============================================================================
# Column 2: Online Inference (middle)
# ============================================================================

ax.text(8, 9.3, 'Online Inference', ha='center', fontsize=12, fontweight='bold', color='#555555')

# 1. User Query
draw_box(6.5, 8, 3, 0.8, 'User Query', COLORS['output'])

# 2. Retrieval
draw_box(6.5, 6.5, 3, 0.8, 'Retrieval Mode\\n(dense|sparse|hybrid)', COLORS['chunk'])
draw_box(6.5, 5.2, 3, 0.9, 'Top-k Retrieval\\n(cosine similarity)', COLORS['chunk'])

# 3. Prompt Builder
draw_box(6.5, 3.8, 3, 0.8, 'Prompt Builder\\n(rag_chatbot.py)', COLORS['prompt'])

# ============================================================================
# Column 3: LLM Backends (right)
# ============================================================================

# 1. LLM Interface (Highlight 3)
draw_box(11.5, 6, 3, 1.0, 'LLM Interface\\n(llm_interface.py)', COLORS['llm'], is_highlight=True, fontsize=9)

# 2. Backends
draw_box(10.5, 4.5, 1.6, 0.7, 'HuggingFace\\n(TinyLlama)', COLORS['llm'], fontsize=8)
draw_box(12.2, 4.5, 1.6, 0.7, 'Ollama', COLORS['llm'], fontsize=8)
draw_box(13.9, 4.5, 1.6, 0.7, 'OpenAI\\nCompatible', COLORS['llm'], fontsize=8)

# 3. Answer
draw_box(11.5, 2.5, 3, 0.8, 'Grounded Answer', COLORS['output'])

# ============================================================================
# Arrows: Offline
# ============================================================================

# Data -> Chunk
draw_arrow(2.5, 7.9, 2.5, 7.7)

# Chunk -> Enrich & Embed
draw_arrow(1.9, 6.6, 1.4, 6.3)
draw_arrow(2.95, 6.4, 2.95, 5.0)

# Enrich -> Embed
draw_arrow(2.25, 5.5, 2.95, 5.2)

# Embed -> Index
draw_arrow(2.95, 4.1, 2.5, 2.7)
draw_arrow(2.95, 3.1, 2.5, 2.7)

# ============================================================================
# Arrows: Online
# ============================================================================

# Query -> Retrieval mode
draw_arrow(8, 7.9, 8, 7.3)
draw_arrow(8, 6.4, 8, 6.1)

# Index -> Retrieval (dashed blue)
draw_arrow(4, 2.25, 6.5, 5.65, color='#0066cc', linestyle='--')

# Retrieval -> Prompt
draw_arrow(8, 5.1, 8, 4.6)

# Prompt -> LLM
draw_arrow(9.5, 4.2, 11.5, 6.5)

# LLM -> Backends
draw_arrow(12.3, 5.9, 11.3, 5.2)
draw_arrow(13, 5.9, 13, 5.2)
draw_arrow(13.7, 5.9, 14.7, 5.2)

# Backends -> Answer
draw_arrow(11.3, 4.4, 13, 3.3)
draw_arrow(13, 4.4, 13, 3.3)
draw_arrow(14.7, 4.4, 13, 3.3)

# ============================================================================
# Baseline bypass (red dotted)
# ============================================================================
draw_arrow(9.5, 8.4, 9.5, 8.7, color='#ff4444', linestyle=':', linewidth=2)
draw_arrow(9.5, 8.7, 14.5, 8.7, color='#ff4444', linestyle=':', linewidth=2)
draw_arrow(14.5, 8.7, 14.5, 3.5, color='#ff4444', linestyle=':', linewidth=2)
draw_arrow(14.5, 3.5, 14.5, 3.3, color='#ff4444', linestyle=':', linewidth=2)
ax.text(14.8, 6, 'Baseline\\n(no retrieval)', ha='left', va='center', 
        fontsize=9, color='#ff4444', fontweight='bold')

# ============================================================================
# Contribution callouts
# ============================================================================

contrib_y = 0.8
ax.text(2.5, contrib_y, '1. Metadata-enriched chunking', ha='center', 
        fontsize=10, color='#cc0000', fontweight='bold')
ax.text(8, contrib_y, '2. Hybrid dense+sparse retrieval', ha='center', 
        fontsize=10, color='#cc0000', fontweight='bold')
ax.text(13, contrib_y, '3. Multi-backend LLM abstraction', ha='center', 
        fontsize=10, color='#cc0000', fontweight='bold')

# Title
ax.text(8, 9.8, 'Shakespeare SLM/RAG System Architecture', ha='center', 
        fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('c:\\Users\\ZhuanZ\\William_Shakespeare_SLM_RAG\\report\\architecture_diagram.png', 
            dpi=200, bbox_inches='tight', facecolor='white')
print("Architecture diagram saved to: report/architecture_diagram.png")

plt.show()
