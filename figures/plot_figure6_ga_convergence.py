"""
Figure 6: GA Convergence Curve
Data loaded from CSV, no hard-coded values
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 1.5

CSV_PATH = Path(__file__).parent.parent / "data" / "csv" / "figure6_ga_convergence.csv"


def load_convergence_data(csv_path):
    generations = []
    all_runs = {f"Run_{i}": [] for i in range(1, 11)}
    mean_fitness = []
    std_fitness = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            generations.append(int(row["Generation"]))
            for i in range(1, 11):
                all_runs[f"Run_{i}"].append(float(row[f"Run_{i}"]))
            mean_fitness.append(float(row["Mean"]))
            std_fitness.append(float(row["Std"]))

    generations = np.array(generations)
    mean_fitness = np.array(mean_fitness)
    std_fitness = np.array(std_fitness)
    all_runs_array = np.array([all_runs[f"Run_{i}"] for i in range(1, 11)])

    return generations, all_runs_array, mean_fitness, std_fitness


def create_figure():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {CSV_PATH}")

    generations, all_runs_array, mean_fitness, std_fitness = load_convergence_data(CSV_PATH)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(10):
        ax.plot(generations, all_runs_array[i], color='gray', alpha=0.15, linewidth=0.8)

    ax.plot(generations, mean_fitness, color='#1f77b4', linewidth=3,
            label='Mean Fitness (10 runs)')

    ax.fill_between(generations, mean_fitness - std_fitness, mean_fitness + std_fitness,
                    color='#1f77b4', alpha=0.15, label=r'Inter-run Variance ($\pm 1\sigma$)')

    convergence_value = 15
    ax.axhline(y=convergence_value, color='#d62728', linestyle='--', linewidth=1.5,
               alpha=0.8, label='Global Optimal Basin')

    ax.set_xlabel('Generations', fontsize=14, weight='bold')
    ax.set_ylabel('Fitness Value (Loss)', fontsize=14, weight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 120)

    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 20, 40, 60, 80, 100, 120])

    ax.legend(loc='upper right', framealpha=0.95, edgecolor='black',
              fontsize=11, fancybox=False)

    ax.grid(True, linestyle=':', alpha=0.4, color='gray')

    plt.tight_layout()

    output_prefix = str(Path(__file__).resolve().parent / 'Figure6_GA_Convergence_FromCSV')
    plt.savefig(f'{output_prefix}.pdf', dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{output_prefix}.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{output_prefix}.svg', format='svg', bbox_inches='tight', facecolor='white')

    print(f"Figure 6 generated from CSV: {CSV_PATH}")


if __name__ == "__main__":
    create_figure()
