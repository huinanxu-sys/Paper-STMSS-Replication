"""
Figure 1: Lévy vs Gaussian Distribution
Demonstrates the Lévy distribution characteristics of mosquito flight paths
compared to Gaussian distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
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
    
    ax1.set_xlabel('Step Length', fontweight='bold')
    ax1.set_ylabel('Probability Density', fontweight='bold')
    ax1.set_title('(a) Step Length Distribution', fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2 = axes[1]
    x_tail = np.linspace(0.1, 10, 500)
    
    gaussian_tail = norm.pdf(x_tail, 0, 1)
    levy_15_tail = levy_stable.pdf(x_tail, alpha=1.5, beta=0, loc=0, scale=1)
    
    ax2.loglog(x_tail, gaussian_tail, color='#1f77b4', linewidth=2.5,
               label='Gaussian (α=2.0)')
    ax2.loglog(x_tail, levy_15_tail, color='#ff7f0e', linewidth=2.5,
               label='Lévy Stable (α=1.5)')
    
    x_ref = np.array([0.5, 10])
    y_ref = 0.3 * x_ref**(-2.5)
    ax2.loglog(x_ref, y_ref, 'k--', linewidth=1.5, alpha=0.5,
               label=r'Power Law ($\propto x^{-2.5}$)')
    
    ax2.set_xlabel('Step Length (log scale)', fontweight='bold')
    ax2.set_ylabel('Probability Density (log scale)', fontweight='bold')
    ax2.set_title('(b) Heavy-Tailed Behavior', fontweight='bold')
    ax2.legend(loc='lower left', framealpha=0.95, fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.suptitle('Figure 1: Lévy vs Gaussian Distribution in Vector Flight',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    plt.savefig('Figure1_Levy_Distribution.pdf',
                bbox_inches='tight', facecolor='white')
    plt.savefig('Figure1_Levy_Distribution.png',
                bbox_inches='tight', facecolor='white')
    
    print("Figure 1 generated: Lévy Distribution")


if __name__ == "__main__":
    create_figure1()
