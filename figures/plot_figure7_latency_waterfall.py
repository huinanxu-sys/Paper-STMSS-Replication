"""
Figure 7: End-to-End Cyber-Physical Latency Waterfall (5 pipelines)

Shows total cyber-physical latency (edge inference + 25.0 ms mechanical
solenoid delay) versus the 75.0 ms actuation deadline for all five
pipelines evaluated in the manuscript:

    * YOLOv8n + ByteTrack    (semantic DL baseline)
    * MOG2 + Lucas-Kanade    (lightweight non-semantic baseline)
    * MOG2 + SORT            (lightweight non-semantic baseline)
    * MOG2 + IMM (2-model)   (non-linear control-theory baseline)
    * STMSS (Proposed)       (offline-evolved linear KF)

Data source: data/csv/figure7_latency_data.csv
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "csv" / "figure7_latency_data.csv"


def load_latency_data(csv_path):
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'system':   row['System'],
                'infer':    float(row['Inference_Latency_ms']),
                'mech':     float(row['Mechanical_Latency_ms']),
                'total':    float(row['Total_Latency_ms']),
                'deadline': float(row['Deadline_ms']),
                'margin':   float(row['Margin_ms']),
                'status':   row['Status'],
            })
    return rows


def status_color(status):
    return {
        'GUARANTEED': '#059669',
        'MARGINAL':   '#f59e0b',
        'FAIL':       '#ef4444',
    }[status]


def create_figure7():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {CSV_PATH}")

    rows = load_latency_data(CSV_PATH)
    systems = [r['system'] for r in rows]
    inference = [r['infer'] for r in rows]
    mechanical = [r['mech'] for r in rows]
    totals = [r['total'] for r in rows]
    status_colors = [status_color(r['status']) for r in rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))

    y_pos = np.arange(len(systems))
    bar_height = 0.55

    color_comp = '#cbd5e1'
    color_mech = '#64748b'
    color_deadline = '#ef4444'

    ax.barh(y_pos, inference, height=bar_height,
            color=color_comp, edgecolor='#334155',
            label=r'Edge Inference Latency ($T_{comp}$)')
    ax.barh(y_pos, mechanical, left=inference, height=bar_height,
            color=color_mech, edgecolor='#334155',
            label=r'Mechanical Valve Delay ($T_{mech}$ = 25.0 ms)')

    ax.axvline(x=75.0, color=color_deadline, linestyle='--', linewidth=2.5,
               label=r'Actuation Deadline ($T_{allow}$ = 75.0 ms)')

    # Right-edge status indicator
    for i, (t, c) in enumerate(zip(totals, status_colors)):
        ax.plot(t, i, marker='o', markersize=9, color=c, zorder=5,
                markeredgecolor='white', markeredgewidth=1.5)

    # Per-row status text
    for i, r in enumerate(rows):
        label = 'GUARANTEED' if r['status'] == 'GUARANTEED' else r['status']
        ax.text(76.5, i, f"{label}\n({r['margin']:+.2f} ms)",
                color=status_color(r['status']), fontweight='bold',
                va='center', ha='left', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(systems, fontweight='bold')
    ax.set_xlabel('End-to-End Cyber-Physical Latency (ms)', fontweight='bold')
    ax.set_xlim(0, 105)

    ax.legend(loc='lower right', framealpha=0.95, fontsize=9)
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    out_dir = Path(__file__).resolve().parent
    plt.savefig(out_dir / 'Figure7_Latency_Waterfall.pdf',
                bbox_inches='tight', facecolor='white')
    plt.savefig(out_dir / 'Figure7_Latency_Waterfall.png',
                bbox_inches='tight', facecolor='white', dpi=600)
    plt.savefig(out_dir / 'Figure7_Latency_Waterfall.svg',
                bbox_inches='tight', facecolor='white')

    print(f"Figure 7 generated from CSV: {CSV_PATH}")


if __name__ == "__main__":
    create_figure7()
