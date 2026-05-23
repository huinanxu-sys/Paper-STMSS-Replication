"""
Figure 9: PAD Workload Distribution
Pneumatic Actuation Debounce (PAD) workload reduction effect

Data source: 04_Data_GroundTruth/figure9_pad_workload.csv
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

CSV_PATH = Path(__file__).parent.parent / "04_Data_GroundTruth" / "figure9_pad_workload.csv"


def load_workload_data(csv_path):
    frames = []
    original = []
    filtered = []
    reduction = []
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row['Frame']))
            original.append(int(row['Original_Detections']))
            filtered.append(int(row['Filtered_Detections']))
            reduction.append(float(row['Reduction_Percent']))
    
    return np.array(frames), np.array(original), np.array(filtered), np.array(reduction)


def create_figure9():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {CSV_PATH}")
    
    frames, original, filtered, reduction = load_workload_data(CSV_PATH)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), 
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.fill_between(frames, original, alpha=0.3, color='#ef4444', 
                     label='Original Detections (w/o PAD)')
    ax1.fill_between(frames, filtered, alpha=0.5, color='#059669',
                     label='Filtered Detections (w/ PAD)')
    ax1.plot(frames, original, color='#ef4444', linewidth=1.5, linestyle='--')
    ax1.plot(frames, filtered, color='#059669', linewidth=2)
    
    ax1.set_ylabel('Number of Detections', fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2.bar(frames, reduction, width=30, color='#3b82f6', alpha=0.7,
            edgecolor='#1e40af', linewidth=1)
    ax2.axhline(y=np.mean(reduction), color='#ef4444', linestyle='--', 
                linewidth=2, label=f'Mean Reduction: {np.mean(reduction):.1f}%')
    
    ax2.set_xlabel('Frame Number', fontweight='bold')
    ax2.set_ylabel('Reduction (%)', fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.95, fontsize=9)
    ax2.grid(axis='y', linestyle=':', alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    plt.savefig('Figure9_PAD_Workload.pdf',
                bbox_inches='tight', facecolor='white')
    plt.savefig('Figure9_PAD_Workload.png',
                bbox_inches='tight', facecolor='white')

    print(f"Figure 9 generated from CSV: {CSV_PATH}")


if __name__ == "__main__":
    create_figure9()
