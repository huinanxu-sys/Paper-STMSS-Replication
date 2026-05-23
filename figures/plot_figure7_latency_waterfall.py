"""
Figure 7: Cyber-Physical Actuation Bound & Latency Confinement
Latency waterfall comparison at 18.82ms vs 75ms deadline

Data source: 04_Data_GroundTruth/figure7_latency_data.csv
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

CSV_PATH = Path(__file__).parent.parent / "04_Data_GroundTruth" / "figure7_latency_data.csv"


def load_latency_data(csv_path):
    systems = []
    inference = []
    mechanical = []
    total = []
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            systems.append(row['System'].replace('_', '\n'))
            inference.append(float(row['Inference_Latency_ms']))
            mechanical.append(float(row['Mechanical_Latency_ms']))
            total.append(float(row['Total_Latency_ms']))
    
    return systems, inference, mechanical, total


def create_figure7():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {CSV_PATH}")
    
    systems, inference, mechanical, total = load_latency_data(CSV_PATH)
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    y_pos = np.arange(len(systems))
    bar_height = 0.5
    
    color_comp = '#cbd5e1'
    color_mech = '#64748b'
    color_deadline = '#ef4444'
    
    ax.barh(y_pos, inference, height=bar_height, 
            color=color_comp, edgecolor='#334155',
            label=r'Edge Inference Latency ($T_{comp}$)')
    ax.barh(y_pos, mechanical, left=inference, height=bar_height,
            color=color_mech, edgecolor='#334155',
            label=r'Mechanical Valve Delay ($T_{mech}$)')
    
    ax.axvline(x=75.0, color=color_deadline, linestyle='--', linewidth=2.5,
               label=r'Actuation Deadline ($T_{allow} = 75.0$ ms)')
    
    ax.text(73.0, 0, 'MARGINAL / FAIL\n(Breaches Bound)', 
            color=color_deadline, fontweight='bold', va='center', 
            ha='right', fontsize=10)
    ax.text(45.5, 1, 'GUARANTEED\n(31.18 ms Margin)', 
            color='#059669', fontweight='bold', va='center', 
            ha='left', fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(systems, fontweight='bold')
    ax.set_xlabel('End-to-End Cyber-Physical Latency (ms)', fontweight='bold')
    ax.set_xlim(0, 90)
    
    ax.legend(loc='lower right', framealpha=0.95, fontsize=9)
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    plt.savefig('Figure7_Latency_Waterfall.pdf',
                bbox_inches='tight', facecolor='white')
    plt.savefig('Figure7_Latency_Waterfall.png',
                bbox_inches='tight', facecolor='white')

    print(f"Figure 7 generated from CSV: {CSV_PATH}")


if __name__ == "__main__":
    create_figure7()
