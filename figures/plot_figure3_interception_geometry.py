"""
Figure 3: Geometric and Kinematic Schematic of Cyber-Physical Pneumatic Interception

Compact, zoomed-in layout with large readable text. Three vertically-stacked
panels:
    1. Top:    Geometric schematic (zoomed in)
    2. Middle: YELLOW focal box  (75.0 ms worst-case radial)
    3. Bottom: GRAY reference box (150 ms full diametric, not used)

Output:
    figures/Figure3_Interception_Geometry.png
    figures/Figure3_Interception_Geometry.pdf
    figures/Figure3_Interception_Geometry.svg
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, FancyArrowPatch

# Physical constants
R = 0.15          # 15 cm effective actuation radius
V_ESC = 2.0       # 2 m/s critical sprint velocity
T_ALLOW = R / V_ESC * 1000.0         # 75.0 ms (strict radial)
T_DIAMETRIC = 2 * R / V_ESC * 1000.0 # 150 ms (full diametric)

# ---- Font sizes ----
FS_CAPTURE  = 17    # Capture-zone label
FS_ARROW    = 18    # R / v_esc labels
FS_NOZZLE   = 13    # Nozzle label
FS_PHI      = 17    # Greek letter phi
FS_PHI_CAP  = 13    # phi explanation caption
FS_BOX_HEAD = 17    # Box header
FS_BOX_FORM = 16    # Box formula
FS_BOX_FOOT = 13    # Box footnote
FS_REF      = 14    # Reference box text

# ---- Colour palette ----
COL_CAPTURE = '#dc2626'   # red
COL_R       = '#1e293b'   # dark slate
COL_VESC    = '#dc2626'   # red
COL_NOZZLE  = '#1e293b'   # dark slate
COL_PHI     = '#475569'   # slate
COL_FOCAL   = '#d97706'   # amber (yellow box border)
COL_REF     = '#94a3b8'   # gray (reference box border)


def create_figure3():
    # ---- Single, more square figure (zoomed in) ----
    fig = plt.figure(figsize=(9.5, 10.0))

    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        height_ratios=[3.4, 1.0, 0.65],   # geometry : focal-box : ref-box
        hspace=0.22,
        left=0.05, right=0.95,
        top=0.97, bottom=0.04,
    )

    # =========================================================================
    # Panel 1 (top): Geometric schematic -- zoomed in
    # =========================================================================
    ax = fig.add_subplot(gs[0])
    ax.set_aspect('equal')
    ax.set_xlim(-0.22, 0.46)
    ax.set_ylim(-0.20, 0.20)
    ax.axis('off')

    # Capture-zone circle (radius R)
    circle = Circle((0, 0), R, fill=False, edgecolor=COL_CAPTURE,
                    linewidth=3.0, linestyle='-')
    ax.add_patch(circle)
    # Label placed at top of circle
    ax.text(0, 0.190, r'Capture zone  (R = 15 cm)',
            fontsize=FS_CAPTURE, color=COL_CAPTURE, fontweight='bold',
            ha='center', va='bottom')

    # Wedge showing the radial penetration path
    wedge = Wedge((0, 0), R, 175, 185, facecolor='#fef3c7',
                  edgecolor=COL_FOCAL, linewidth=2.0, alpha=0.75)
    ax.add_patch(wedge)

    # ---- Incoming target arrow (ABOVE radial axis, no overlap) ----
    arr_vesc = FancyArrowPatch(
        (R + 0.08, 0.090), (0.02, 0.090),
        arrowstyle='-|>', mutation_scale=24,
        color=COL_VESC, linewidth=3.0,
    )
    ax.add_patch(arr_vesc)
    ax.text(R + 0.09, 0.090, r'$v_{\mathrm{esc}}$ = 2.0 m/s',
            fontsize=FS_ARROW, color=COL_VESC, fontweight='bold',
            va='center')

    # ---- Air-knife nozzle (apex) at origin ----
    ax.scatter([0], [0], s=200, c='#1e293b', marker='s', zorder=5)
    # Nozzle label placed to the LEFT and slightly BELOW (in a clear
    # empty zone, no overlap with R label which sits BELOW the x-axis)
    ax.annotate('Air-knife\nnozzle apex',
                xy=(-0.018, -0.005), xytext=(-0.20, -0.115),
                fontsize=FS_NOZZLE, ha='center', va='center', color=COL_NOZZLE,
                fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=COL_NOZZLE, lw=1.0))

    # ---- Radial penetration arrow (R) on the x-axis ----
    arr_R = FancyArrowPatch(
        (0.005, 0), (R, 0),
        arrowstyle='<|-|>',
        mutation_scale=20,
        color=COL_R, linewidth=2.0,
    )
    ax.add_patch(arr_R)
    # R label placed BELOW the radial axis, but at y=-0.040 (clear of nozzle
    # label which is at y=-0.115)
    ax.text(R / 2, -0.040, r'R = 15 cm  (radial penetration)',
            fontsize=FS_ARROW, color=COL_R, ha='center', va='top',
            fontweight='bold')

    # ---- Phi angle: small arc + symbol ABOVE the x-axis near the nozzle,
    # and explanation caption placed BELOW the circle (clear of all labels) ----
    phi_arc = Wedge((0, 0), R * 0.22, 0, 8, facecolor='#cbd5e1',
                    edgecolor=COL_PHI, linewidth=1.0, alpha=0.7)
    ax.add_patch(phi_arc)
    ax.text(0.045, 0.022, r'$\phi$', fontsize=FS_PHI,
            color=COL_PHI, fontstyle='italic', va='bottom')

    # =========================================================================
    # Panel 2 (middle): YELLOW focal box  (WORST-CASE RADIAL, 75.0 ms)
    # =========================================================================
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    box_text = (
        r'$\mathbf{WORST\!-\!CASE\ RADIAL}$'
        r'   (this work uses this bound)' '\n'
        r'$T_{\mathrm{allow}} = R\,/\,v_{\mathrm{esc}}'
        r' = 15\,\mathrm{cm}\;/\;2.0\,\mathrm{m/s}'
        r' = \mathbf{75.0\ \mathbf{ms}}$' '\n'
        r'(strict radial upper bound, R = 15 cm, v_esc = 2.0 m/s)'
    )
    ax2.text(0.5, 0.5, box_text,
             fontsize=FS_BOX_FORM, ha='center', va='center',
             linespacing=1.7,
             bbox=dict(boxstyle='round,pad=1.0',
                       facecolor='#fef3c7', edgecolor=COL_FOCAL,
                       linewidth=2.8))

    # =========================================================================
    # Panel 3 (bottom): GRAY reference box  (150 ms, not used)
    # =========================================================================
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    ref_text = (
        r'For reference: full diametric traversal  '
        r'$T_{\mathrm{diam}} = 2R\,/\,v_{\mathrm{esc}} = 150\ \mathrm{ms}$  '
        r'(not used in this work)'
    )
    ax3.text(0.5, 0.5, ref_text,
             fontsize=FS_REF, ha='center', va='center',
             color='#475569', style='italic',
             bbox=dict(boxstyle='round,pad=0.5',
                       facecolor='#f1f5f9', edgecolor=COL_REF,
                       linewidth=1.2))

    # ---- No title (EAAI supplies the caption separately) ----

    # ---- Save ----
    out_dir = Path(__file__).resolve().parent
    for ext in ('png', 'pdf', 'svg'):
        out_path = out_dir / f'Figure3_Interception_Geometry.{ext}'
        plt.savefig(out_path, bbox_inches='tight', facecolor='white', dpi=600)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    create_figure3()
