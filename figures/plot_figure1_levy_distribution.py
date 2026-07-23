"""
Figure 1: Heavy-Tailed Innovation Sequence vs. Gaussian Noise Model.

The tracker's *innovation sequence* (prediction-error displacement) is modelled
via a Truncated Lévy-Stable distribution instead of a traditional Gaussian
profile. The highlighted heavy-tail region (High-Velocity Saccade Regime)
strictly accounts for the extreme biological micro-saccades and abrupt aerial
turns observed in the innovation (residual) signal, which standard
normally-distributed models mathematically neglect as outlier noise, leading to
critical physical containment failures.

The figure is intentionally *not* labelled "Lévy flight trajectory" because
the macroscopic biological trajectory is constrained by mass-inertia limits
to near-linear motion; only the innovation (residual) sequence exhibits the
heavy-tailed Lévy-Stable statistics.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import norm, levy_stable

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600


def create_figure1():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    x = np.linspace(-5, 5, 1000)

    gaussian = norm.pdf(x, 0, 1)
    ax1.plot(x, gaussian, color='#1f77b4', linewidth=2.5,
             label='Gaussian (α=2.0)')

    levy_15 = levy_stable.pdf(x, alpha=1.5, beta=0, loc=0, scale=1)
    ax1.plot(x, levy_15, color='#ff7f0e', linewidth=2.5,
             label='Lévy Stable (α=1.5)')

    levy_10 = levy_stable.pdf(x, alpha=1.0, beta=0, loc=0, scale=1)
    ax1.plot(x, levy_10, color='#2ca02c', linewidth=2.5,
             label='Lévy Stable (α=1.0)')

    ax1.set_xlabel('Innovation (Prediction-Error) Displacement', fontweight='bold')
    ax1.set_ylabel('Probability Density', fontweight='bold')
    ax1.set_title('(a) Innovation Distribution', fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Saccade regime highlight
    ax1.axvspan(2.0, 5.0, color='#ff7f0e', alpha=0.10,
                label='Saccade regime (heavy tail)')
    ax1.text(2.5, 0.25, 'High-Velocity\nSaccade Regime',
             color='#c2410c', fontweight='bold', fontsize=9,
             ha='left', va='center')

    ax2 = axes[1]
    x_tail = np.linspace(0.1, 10, 500)

    gaussian_tail = norm.pdf(x_tail, 0, 1)
    levy_15_tail = levy_stable.pdf(x_tail, alpha=1.5, beta=0, loc=0, scale=1)

    ax2.loglog(x_tail, gaussian_tail, color='#1f77b4', linewidth=2.5,
               label='Gaussian (α=2.0)')
    ax2.loglog(x_tail, levy_15_tail, color='#ff7f0e', linewidth=2.5,
               label='Lévy Stable (α=1.5)')

    x_ref = np.array([0.5, 10])
    y_ref = 0.3 * x_ref ** (-2.5)
    ax2.loglog(x_ref, y_ref, 'k--', linewidth=1.5, alpha=0.5,
               label=r'Power Law ($\propto x^{-2.5}$)')

    ax2.set_xlabel('Innovation Displacement (log scale)', fontweight='bold')
    ax2.set_ylabel('Probability Density (log scale)', fontweight='bold')
    ax2.set_title('(b) Heavy-Tailed Behaviour of the Innovation Sequence',
                  fontweight='bold')
    ax2.legend(loc='lower left', framealpha=0.95, fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.suptitle('Figure 1: Heavy-Tailed Innovation Sequence vs. Gaussian Noise',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    out_dir = Path(__file__).resolve().parent
    plt.savefig(out_dir / 'Figure1_Levy_Distribution.pdf',
                bbox_inches='tight', facecolor='white')
    plt.savefig(out_dir / 'Figure1_Levy_Distribution.png',
                bbox_inches='tight', facecolor='white', dpi=600)
    plt.savefig(out_dir / 'Figure1_Levy_Distribution.svg',
                bbox_inches='tight', facecolor='white')

    print("Figure 1 generated: Heavy-Tailed Innovation Sequence")


if __name__ == "__main__":
    create_figure1()
