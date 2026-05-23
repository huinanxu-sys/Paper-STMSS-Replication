"""
Figure 8: Monte Carlo Sensitivity Analysis of P_capture vs T_comp
System ensures P_capture >= 95.2% at 18.82 ms latency

Data source: 04_Data_GroundTruth/figure8_survival_probability.csv
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path
from scipy.interpolate import make_interp_spline

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

CSV_PATH = Path(__file__).parent.parent / "04_Data_GroundTruth" / "figure8_survival_probability.csv"


def load_survival_data(csv_path):
    t_comp = []
    p_capture = []
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_comp.append(float(row['T_comp_ms']))
            p_capture.append(float(row['P_capture_percent']))
    
    return np.array(t_comp), np.array(p_capture)


def create_figure8():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {CSV_PATH}")
    
    t_comp, p_capture = load_survival_data(CSV_PATH)
    
    fig, ax = plt.subplots(figsize=(8, 4.8))
    
    x_smooth = np.linspace(t_comp.min(), t_comp.max(), 500)
    spl = make_interp_spline(t_comp, p_capture, k=3)
    y_smooth = spl(x_smooth)
    
    ax.plot(x_smooth, y_smooth, color='#1e293b', linewidth=2.5,
            label=r'Simulated $P_{capture}$ (Lévy-Stable Environment)')
    
    ax.axhline(y=95.0, color='#64748b', linestyle=':', linewidth=1.5,
               label='95% Biosecurity Safety Threshold')
    
    stmss_idx = np.argmin(np.abs(t_comp - 18.82))
    ax.axvline(x=18.82, color='#059669', linestyle='--', linewidth=2.0)
    ax.plot(18.82, p_capture[stmss_idx], marker='o', markersize=8,
            color='#059669',
            label=rf'STMSS ($T_{{comp}} = 18.82$ ms, $P_{{capture}} = {p_capture[stmss_idx]:.2f}\%$)')
    
    ax.axvline(x=50.0, color='#ef4444', linestyle='--', linewidth=2.0)
    ax.text(47.5, 50, r'Computational Limit ($T_{comp} = 50.0$ ms)',
            color='#ef4444', rotation=90, va='center', ha='center',
            fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Computational Latency $T_{comp}$ (ms)', fontweight='bold')
    ax.set_ylabel('Capture Probability $P_{capture}$ (%)', fontweight='bold')
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 105)
    
    ax.legend(loc='lower left', framealpha=0.95, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    plt.savefig('Figure8_Survival_Probability.pdf',
                bbox_inches='tight', facecolor='white')
    plt.savefig('Figure8_Survival_Probability.png',
                bbox_inches='tight', facecolor='white')

    print(f"Figure 8 generated from CSV: {CSV_PATH}")


if __name__ == "__main__":
    create_figure8()
