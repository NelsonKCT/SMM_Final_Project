import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as patches

# Setup style for top-tier academic visualization
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300

# Directory configuration
figures_dir = r"c:\Users\minelab\Desktop\projects\ssm\poster\figures"
os.makedirs(figures_dir, exist_ok=True)

# Aesthetic Palette (Cool Tech / Academic Theme)
COLOR_GREY = '#BDC3C7'      # Silver (Baseline)
COLOR_BLUE = '#34495E'      # Dark Slate Blue (DFA Baseline)
COLOR_ORANGE = '#E67E22'    # Soft Orange (GNN + NHS + TAVS)
COLOR_TEAL = '#1ABC9C'      # Emerald Teal (CS-DFA)
COLOR_RED = '#E74C3C'       # Crimson Red (nogate)

# ----------------------------------------------------
# CHART 1: Results Comparison (Grouped Bar Chart)
# ----------------------------------------------------
countries = ["China", "Iran", "UAE", "Cuba", "Russia", "Venezuela"]
paper_baseline = [0.581, 0.728, 0.839, 0.899, 0.798, 0.910]
dfa_baseline = [0.603, 0.706, 0.880, 0.856, 0.836, 0.768]
gnn_nhs_tavs = [0.614, 0.713, 0.893, 0.884, 0.907, 0.831]
cs_dfa = [0.837, 0.998, 0.993, 0.888, 1.000, 0.998]

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
x = np.arange(len(countries))
width = 0.18

# Draw bars with premium edges and styling
rects1 = ax.bar(x - 1.5*width, paper_baseline, width, label='Paper Baseline (AAAI\'25)', color=COLOR_GREY, edgecolor='none', alpha=0.9)
rects2 = ax.bar(x - 0.5*width, dfa_baseline, width, label='DFA Baseline', color=COLOR_BLUE, edgecolor='none', alpha=0.9)
rects3 = ax.bar(x + 0.5*width, gnn_nhs_tavs, width, label='GNN + NHS + TAVS (Ours)', color=COLOR_ORANGE, edgecolor='none', alpha=0.9)
rects4 = ax.bar(x + 1.5*width, cs_dfa, width, label='CS-DFA (Ours, Best)', color=COLOR_TEAL, edgecolor='none', alpha=0.95)

# Formatting
ax.set_ylabel('Zero-Shot Test F1-Macro', fontsize=12, fontweight='bold', labelpad=8)
ax.set_title('Cross-Country Zero-Shot Transfer Performance Comparison', fontsize=14, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(countries, fontsize=11, fontweight='bold')
ax.set_ylim(0.0, 1.15)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# Add values above bars for CS-DFA (Green)
for rect in rects4:
    height = rect.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),  
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, color='#0E6251', weight='bold')

# Add values above bars for Paper Baseline
for rect in rects1:
    height = rect.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),  
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, color='#5D6D7E')

ax.legend(loc='upper left', frameon=True, edgecolor='#E5E8E8', fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "poster_results_comparison.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(figures_dir, "poster_results_comparison.png"), bbox_inches='tight', dpi=300)
plt.close()

# ----------------------------------------------------
# CHART 2: CS-DFA Ablation (Horizontal Bar Chart for Cleaner Layout)
# ----------------------------------------------------
variants = ["CS-DFA (Full)", "No CORAL", "No Prior (lambda=0)", "No Gating (nogate)", "DFA Baseline"]
avg_f1s = [0.952, 0.950, 0.952, 0.686, 0.775]
colors_ablation = [COLOR_TEAL, '#48C9B0', '#76D7C4', COLOR_RED, COLOR_BLUE]

fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
y_pos = np.arange(len(variants))
bars = ax.barh(y_pos, avg_f1s, color=colors_ablation, height=0.55, edgecolor='none', alpha=0.9)

ax.set_yticks(y_pos)
ax.set_yticklabels(variants, fontsize=10, fontweight='bold')
ax.invert_yaxis()  # top-down
ax.set_xlabel('6-Country Average Test F1-Macro', fontsize=11, fontweight='bold', labelpad=8)
ax.set_title('Ablation Study: Gating is the Key Driver', fontsize=12, fontweight='bold', pad=10)
ax.set_xlim(0.0, 1.1)
ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# Add labels to the right of each bar
for bar in bars:
    width = bar.get_width()
    ax.annotate(f'{width:.3f}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),  
                textcoords="offset points",
                ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "csdfa_ablation_comparison.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(figures_dir, "csdfa_ablation_comparison.png"), bbox_inches='tight', dpi=300)
plt.close()

# ----------------------------------------------------
# CHART 3: System Architecture Diagram (Beautiful Custom Matplotlib Layout)
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Design tokens
FC_INPUT = '#EAECEE'
FC_PROJ = '#D5F5E3'
FC_GNN = '#E8F8F5'
FC_GNN_NOISE = '#FDEDEC'
FC_MASK = '#FEF9E7'
FC_FUSION = '#FCF3CF'
FC_OUTPUT = '#EBDEF0'

# Helper function to draw custom card boxes with shadows
def draw_card(ax, x, y, w, h, text, fill_color, border_color='#34495E', text_color='#2C3E50', fontsize=9, title=""):
    # Draw drop shadow
    shadow = patches.FancyBboxPatch((x+0.05, y-0.05), w, h, boxstyle="round,pad=0.08", 
                                    facecolor='#BDC3C7', edgecolor='none', alpha=0.5)
    ax.add_patch(shadow)
    # Draw actual card
    card = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", 
                                  facecolor=fill_color, edgecolor=border_color, linewidth=1.2)
    ax.add_patch(card)
    
    # Title & text spacing
    if title:
        ax.text(x + w/2, y + h - 0.25, title, ha='center', va='center', color=text_color, fontsize=fontsize+1, fontweight='bold')
        ax.text(x + w/2, y + (h-0.3)/2, text, ha='center', va='center', color=text_color, fontsize=fontsize, wrap=True)
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', color=text_color, fontsize=fontsize, fontweight='bold', wrap=True)

# Helper function to draw clean line arrows
def draw_line_arrow(ax, x1, y1, x2, y2, color='#34495E', lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=10, shrinkA=0, shrinkB=0))

# Title
ax.text(5, 5.7, "CS-DFA Model Architecture Flow", ha='center', va='center', fontsize=14, fontweight='bold', color='#1A5276')

# Blocks
# 1. Inputs
draw_card(ax, 0.4, 3.8, 1.8, 1.0, "Node Features\n- Text (SBERT)\n- Degree Structural", FC_INPUT, title="1. Inputs")

# 2. DFA Projection
draw_card(ax, 2.7, 3.8, 2.0, 1.0, "Decoupled Cross-Attention\nProjection Layer", FC_PROJ, title="2. DFA Projection")
draw_line_arrow(ax, 2.2, 4.3, 2.7, 4.3)

# 3. Channel GNNs
channels = ["coRT GNN", "coURL GNN", "hashSeq GNN (Noise)", "fastRT GNN (Noise)", "tweetSim GNN (Noise)"]
colors_ch = [FC_GNN, FC_GNN, FC_GNN_NOISE, FC_GNN_NOISE, FC_GNN_NOISE]
borders_ch = ['#16A085', '#16A085', '#C0392B', '#C0392B', '#C0392B']

for i in range(5):
    y_pos = 3.6 - i*0.8
    draw_card(ax, 5.3, y_pos, 1.8, 0.5, channels[i], colors_ch[i], border_color=borders_ch[i], fontsize=8)
    # Branching arrows from Projector to Channels
    ax.annotate("", xy=(5.3, y_pos + 0.25), xytext=(4.7, 4.3),
                arrowprops=dict(arrowstyle="-|>", color='#7F8C8D', connectionstyle="angle,angleA=0,angleB=-90,rad=3", mutation_scale=8))

# 4. Coverage Mask
draw_card(ax, 2.7, 1.0, 2.0, 0.7, "Binary Coverage Mask\nM(n, c) \u2208 {0, 1}", FC_MASK, border_color='#D35400', fontsize=8)

# 5. Gated Fusion
draw_card(ax, 7.8, 1.6, 1.8, 1.8, "Coverage-Gated\nFusion Layer\n\nAttScore = -\u221E\nif M(n,c) = 0", FC_FUSION, border_color='#D35400', fontsize=8, title="3. Fusion")

# Connections to Fusion
for i in range(5):
    y_pos = 3.6 - i*0.8
    ax.annotate("", xy=(7.8, 2.5), xytext=(7.1, y_pos + 0.25),
                arrowprops=dict(arrowstyle="-|>", color='#7F8C8D', connectionstyle="angle,angleA=0,angleB=90,rad=3", mutation_scale=8))

# Connection from Mask to Fusion
draw_line_arrow(ax, 4.7, 1.35, 7.8, 2.0, color='#D35400')

# 6. Output
draw_card(ax, 7.8, 0.3, 1.8, 0.7, "Classifier MLP\nPrediction Output", FC_OUTPUT, border_color='#8E44AD', fontsize=9)
draw_line_arrow(ax, 8.7, 1.6, 8.7, 1.0)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "poster_architecture_flow.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(figures_dir, "poster_architecture_flow.png"), bbox_inches='tight', dpi=300)
plt.close()

# ----------------------------------------------------
# 4. Empty/Dummy Background & Logos to satisfy LaTeX templates
# ----------------------------------------------------
# Save a clean empty white background image
fig, ax = plt.subplots(figsize=(8, 11))
ax.axis('off')
fig.savefig(os.path.join(figures_dir, "background.pdf"), bbox_inches='tight')
plt.close()

# Save a dummy CU.png
fig, ax = plt.subplots(figsize=(2, 2))
ax.axis('off')
ax.text(0.5, 0.5, "NTU", fontsize=24, fontweight='bold', color='#1A5276', ha='center', va='center')
fig.savefig(os.path.join(figures_dir, "CU.png"), bbox_inches='tight')
plt.close()

# Save a dummy NI.jpg
fig, ax = plt.subplots(figsize=(2, 2))
ax.axis('off')
ax.text(0.5, 0.5, "GFM", fontsize=24, fontweight='bold', color='#E67E22', ha='center', va='center')
fig.savefig(os.path.join(figures_dir, "NI.jpg"), bbox_inches='tight')
plt.close()

print("Poster figures generated successfully in target directory!")
