"""
Figure 4: STMSS Asymmetric Cyber-Physical Architecture
System architecture schematic diagram.

The T_comp label on the diagram is read from the raw Culex_Transit
STMSS latency CSV at render time. T_mech = 25.0 ms is the EXAIR Super
Air Knife manufacturer datasheet constant.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

STMSS_RAW_CSV = (
    Path(__file__).resolve().parent.parent
    / "data" / "csv" / "table1_semantic_baselines.csv"
)


def _read_stmss_tcomp_ms() -> float:
    """Return the STMSS Culex_Transit mean latency (T_comp) read at
    render time from the raw per-frame measurement CSV. The
    T_comp label on the architecture diagram is populated from this
    value.

    Raises:
        FileNotFoundError: if the raw measurement CSV is missing.
        RuntimeError: if the STMSS Culex_Transit row cannot be located.
    """
    if not STMSS_RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw STMSS latency CSV not found: {STMSS_RAW_CSV}. "
            "This file is the single source of truth for the Figure 4 "
            "T_comp label; do not run this script without it."
        )
    header = None
    with open(STMSS_RAW_CSV, "r", newline="") as fh:
        for line in fh:
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            cells = next(csv.reader([line.rstrip("\r\n")]))
            if header is None:
                header = cells
                continue
            row = dict(zip(header, cells))
            if (row.get("Pipeline") == "STMSS"
                    and row.get("Sequence") == "Culex_Transit"):
                return float(row["Mean_ms"])
    raise RuntimeError(
        f"STMSS Culex_Transit row not found in {STMSS_RAW_CSV}"
    )


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
    
    ax.text(5.1, 6.2, rf'$T_{{comp}} \approx {_read_stmss_tcomp_ms():.2f}$ ms',
            ha='center', fontsize=9, color=colors['success'],
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))

    ax.text(12.05, 5.0, r'$T_{mech} = 25.0$ ms',
            ha='center', fontsize=9, color=colors['danger'],
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcccc', alpha=0.5))

    # No embedded title (EAAI supplies the caption separately)

    plt.tight_layout()

    out_dir = Path(__file__).resolve().parent
    plt.savefig(out_dir / 'Figure4_Architecture.pdf',
                bbox_inches='tight', facecolor='white')
    plt.savefig(out_dir / 'Figure4_Architecture.png',
                bbox_inches='tight', facecolor='white', dpi=600)
    plt.savefig(out_dir / 'Figure4_Architecture.svg',
                bbox_inches='tight', facecolor='white')

    print("Figure 4 generated: Architecture Diagram")


if __name__ == "__main__":
    create_figure4()
