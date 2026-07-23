"""
Figure 9: Pneumatic Actuation Debounce (PAD) Workload Reduction.

Grouped bar chart of the actuation command counts before and after
the PAD filter for the five canonical test sequences. Values are
read from ``data/csv/figure9_pad_workload.csv``. The figure has no
embedded title; the caption is supplied in the manuscript.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

THIS_DIR = Path(__file__).resolve().parent
DATA_CSV = THIS_DIR.parent / "data" / "csv" / "figure9_pad_workload.csv"


def _load_workload_csv(csv_path):
    import csv
    seqs, raw, filt, red = [], [], [], []
    with open(csv_path, 'r', newline='') as f:
        for row in csv.DictReader(f):
            seqs.append(row['Sequence'])
            raw.append(int(row['Raw_Triggers']))
            filt.append(int(row['Post_PAD_Triggers']))
            red.append(float(row['Reduction_Percent']))
    return seqs, np.array(raw), np.array(filt), np.array(red)


def create_figure9():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing: {DATA_CSV}")

    sequences, raw, filt, red = _load_workload_csv(DATA_CSV)
    n = len(sequences)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=600)
    x = np.arange(n)
    w = 0.35

    ax.bar(x - w/2, raw,  w, color='#d0d9e8', edgecolor='#555555',
           label='Raw Kinematic Triggers')
    ax.bar(x + w/2, filt, w, color='#3b5998', edgecolor='#111111',
           label='Post-PAD Filtered Triggers')

    for i in range(n):
        ax.text(x[i] - w/2, raw[i]  + max(raw) * 0.015, f"{raw[i]}",
                ha='center', va='bottom', fontsize=9, color='#1f2937')
        ax.text(x[i] + w/2, filt[i] + max(raw) * 0.015, f"{filt[i]}",
                ha='center', va='bottom', fontsize=9, color='#0f172a')
        ax.text(x[i], max(raw[i], filt[i]) * 0.55, f"{red[i]:+.2f}%",
                ha='center', va='center', fontsize=10, weight='bold',
                color='#0f172a',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#fef3c7', edgecolor='#92400e', linewidth=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(sequences, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Actuation Command Count', fontweight='bold')
    ax.set_xlabel('Test Sequence', fontweight='bold')
    ax.set_ylim(0, max(raw) * 1.18)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=9)

    plt.tight_layout()
    for ext, dpi in (('png', 600), ('pdf', None), ('svg', None)):
        plt.savefig(THIS_DIR / f'Figure9_PAD_Workload.{ext}',
                    bbox_inches='tight', facecolor='white', dpi=dpi)
    plt.close(fig)
    print("Figure 9 written (Figure9_PAD_Workload.png/pdf/svg)")


if __name__ == "__main__":
    create_figure9()
