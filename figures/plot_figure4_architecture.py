"""
Figure 4: STMSS Asymmetric Cyber-Physical Architecture
System architecture schematic diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600


def draw_camera_icon(ax, x, y, size=0.4):
    body = Rectangle((x-size*0.6, y-size*0.4), size*1.2, size*0.8,
                     facecolor='#333333', edgecolor='black', linewidth=1.5)
    ax.add_patch(body)
    lens = Circle((x, y), size*0.25, facecolor='#1f77b4', 
                  edgecolor='black', linewidth=2)
    ax.add_patch(lens)
    ax.plot([x, x], [y-size*0.4, y-size*0.7], 'k-', linewidth=2)
    ax.plot([x-size*0.3, x+size*0.3], [y-size*0.7, y-size*0.7], 'k-', linewidth=2)


def draw_database_icon(ax, x, y, size=0.35):
    for i in range(3):
        offset = i * size * 0.25
        ellipse = mpatches.Ellipse((x, y+offset), size*1.2, size*0.4,
                                   facecolor='#e6f3ff', edgecolor='navy', linewidth=1.5)
        ax.add_patch(ellipse)


def create_figure4():
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    colors = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'light': '#f0f0f0'
    }
    
    input_box = FancyBboxPatch((0.3, 7), 2.2, 1.3, boxstyle="round,pad=0.08",
                              facecolor='#e6f3ff', edgecolor=colors['primary'], linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.4, 7.65, '"Dark Warehouse"\nVideo Feed', ha='center', 
            va='center', fontsize=9, fontweight='bold')
    ax.text(1.4, 7.2, r'$I_{raw}(x,y,t)$', ha='center', fontsize=8, family='monospace')
    
    draw_camera_icon(ax, 1.4, 8.8, size=0.35)
    
    stage_a = FancyBboxPatch((3.2, 6.5), 2.8, 2.3, boxstyle="round,pad=0.08",
                            facecolor='#fff4e6', edgecolor=colors['secondary'], linewidth=2)
    ax.add_patch(stage_a)
    ax.text(4.6, 8.4, 'STAGE A', ha='center', fontsize=10, 
            fontweight='bold', color='darkorange')
    ax.text(4.6, 7.9, 'Entropy-Driven\nSignal Restoration', ha='center', fontsize=9)
    ax.text(4.6, 7.3, r'$\hat{I}_{restored} = f_{PI-EOKF}(I_{raw})$',
            ha='center', fontsize=8, family='monospace')
    
    stage_b = FancyBboxPatch((7.0, 6.5), 3.0, 2.3, boxstyle="round,pad=0.08",
                            facecolor='#e6ffe6', edgecolor=colors['success'], linewidth=2)
    ax.add_patch(stage_b)
    ax.text(8.5, 8.4, 'STAGE B', ha='center', fontsize=10,
            fontweight='bold', color='darkgreen')
    ax.text(8.5, 7.9, 'GA-Embedded\nMulti-Target Tracking', ha='center', fontsize=9)
    ax.text(8.5, 7.3, r'$\mathcal{T} = \{(x,y,v_x,v_y)_i\}_{i=1}^{N}$',
            ha='center', fontsize=8, family='monospace')
    
    stage_c = FancyBboxPatch((10.8, 6.5), 2.5, 2.3, boxstyle="round,pad=0.08",
                            facecolor='#ffe6e6', edgecolor=colors['danger'], linewidth=2)
    ax.add_patch(stage_c)
    ax.text(12.05, 8.4, 'STAGE C', ha='center', fontsize=10,
            fontweight='bold', color='darkred')
    ax.text(12.05, 7.9, 'PAD Filter', ha='center', fontsize=9)
    ax.text(12.05, 7.3, r'$\mathcal{T}_{filtered} = f_{PAD}(\mathcal{T})$',
            ha='center', fontsize=8, family='monospace')
    
    output_box = FancyBboxPatch((10.8, 2.5), 2.5, 2.0, boxstyle="round,pad=0.08",
                               facecolor='#f0f0f0', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(12.05, 3.8, 'Pneumatic\nActuation Array', ha='center',
            va='center', fontsize=9, fontweight='bold')
    ax.text(12.05, 3.0, r'$A_{trigger}(t) \in \{0,1\}^{M \times N}$',
            ha='center', fontsize=8, family='monospace')
    
    arrow_style = dict(arrowstyle='->', color='black', lw=2,
                       connectionstyle='arc3,rad=0')
    
    ax.annotate('', xy=(3.2, 7.65), xytext=(2.5, 7.65),
                arrowprops=arrow_style)
    ax.annotate('', xy=(7.0, 7.65), xytext=(6.0, 7.65),
                arrowprops=arrow_style)
    ax.annotate('', xy=(10.8, 7.65), xytext=(10.0, 7.65),
                arrowprops=arrow_style)
    ax.annotate('', xy=(12.05, 4.5), xytext=(12.05, 6.5),
                arrowprops=arrow_style)
    
    ax.text(5.1, 6.2, r'$T_{comp} \approx 18.82$ ms',
            ha='center', fontsize=9, color=colors['success'],
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))
    
    ax.text(12.05, 5.0, r'$T_{mech} = 25.0$ ms',
            ha='center', fontsize=9, color=colors['danger'],
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcccc', alpha=0.5))
    
    ax.text(7, 10.3, 'Figure 4: STMSS Asymmetric Cyber-Physical Architecture',
            ha='center', fontsize=14, fontweight='bold')
    ax.text(7, 9.8, 'Three-Stage Edge Processing Pipeline with Sub-20ms Latency',
            ha='center', fontsize=11, style='italic')
    
    plt.tight_layout()
    
    plt.savefig('Figure4_Architecture.pdf',
                bbox_inches='tight', facecolor='white')
    plt.savefig('Figure4_Architecture.png',
                bbox_inches='tight', facecolor='white')
    
    print("Figure 4 generated: Architecture Diagram")


if __name__ == "__main__":
    create_figure4()
