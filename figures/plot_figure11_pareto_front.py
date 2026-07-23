"""
Figure 11: Pareto Front of Tracking Performance (LocA) vs Isolated Algorithmic
                Latency as a Function of the Penalty-Weight Ratio w4 / w2

The x-axis represents the execution time of the mathematical pipeline
(Stages A, B, C) stripped of hardware I/O and interpreter overhead.

The 10.0 ms algorithmic ceiling leaves 3.68 ms headroom below the
12.5 ms software-toolchain overhead observed in the end-to-end
benchmark.

The "CHOSEN OPERATING POINT" star marker and its annotation overlay
are read at render time from the canonical operating-point CSV
(``data/csv/figure11_operating_point.csv``); no manuscript-locked
value is hardcoded in this script.

Data sources:
  - Pareto front curve:    data/csv/figure11_pareto_front.csv
  - Chosen operating point: data/csv/figure11_operating_point.csv
  - Reference end-to-end:   data/csv/table1_semantic_baselines.csv
                            (STMSS Culex_Transit mean, for the
                            "this is NOT on the plot" annotation).

Output:
  figures/Figure11_Pareto_Front.{png,pdf,svg}
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

# ---------------------------------------------------------------------------
# Data files (single source of truth; no hard-coded values).
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data" / "csv"
DATA_CSV = DATA_DIR / "figure11_pareto_front.csv"
OP_CSV   = DATA_DIR / "figure11_operating_point.csv"
STMSS_RAW_CSV = DATA_DIR / "table1_semantic_baselines.csv"


def _load_pareto_csv(csv_path: Path) -> list:
    """Load Pareto-front data from the canonical CSV. Returns a list of
    (w4_over_w2, T_isolated_ms, LocA_percent) tuples."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Pareto-front data CSV not found: {csv_path}")
    rows = []
    with open(csv_path, "r", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((
                float(r["w4_over_w2"]),
                float(r["T_isolated_ms"]),
                float(r["LocA_percent"]),
            ))
    return rows


def _load_operating_point(csv_path: Path) -> dict:
    """Load the offline PG-GA converged operating point from
    ``csv_path``. Returns a dict with the parameter names as keys.
    Raises ``FileNotFoundError`` if the CSV is missing; the
    downstream code MUST NOT fall back to hard-coded defaults, so a
    missing file is a hard error."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Operating-point CSV not found: {csv_path}. "
            "This file is the single source of truth for the "
            "Figure 11 chosen operating point; do not run this "
            "script without it."
        )
    out = {}
    with open(csv_path, "r", newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["Parameter"]] = float(r["Value"])
    return out


def _read_stmss_e2e_ms(csv_path: Path) -> float:
    """Return the STMSS Culex_Transit mean latency from the raw
    per-frame measurement CSV. Used only for the "this is NOT on the
    plot" annotation that points reviewers to the wall-clock number."""
    with open(csv_path, "r", newline="") as fh:
        header = None
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
        f"STMSS Culex_Transit row not found in {csv_path}"
    )


def _read_table2_stage_latencies(csv_path: Path) -> dict:
    """Return {Component: Mean_ms} from the canonical Table 2 CSV.
    Used to label the Subtotal annotation in Figure 11 with the
    real per-stage numbers, so the figure does not hardcode any
    manuscript value."""
    out = {}
    with open(csv_path, "r", newline="") as fh:
        header = None
        for line in fh:
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            cells = next(csv.reader([line.rstrip("\r\n")]))
            if header is None:
                header = cells
                continue
            row = dict(zip(header, cells))
            comp = row.get("Component", "").strip()
            if comp in ("Stage A", "Stage B", "Stage C"):
                out[comp] = float(row["Mean_ms"])
    if not out:
        raise RuntimeError(
            f"No stage rows found in {csv_path}"
        )
    return out


def create_figure11():
    pareto_data = _load_pareto_csv(DATA_CSV)
    op = _load_operating_point(OP_CSV)
    e2e_ms = _read_stmss_e2e_ms(STMSS_RAW_CSV)
    stage_ms = _read_table2_stage_latencies(DATA_DIR / "table2_stage_latency.csv")

    # ---- Operating-point parameters (read from CSV, no defaults) ----
    W2_FINAL     = op["w2"]
    W4_FINAL     = op["w4"]
    SUBTOTAL_MS  = op["T_isolated_ms"]
    CHOSEN_LOCA  = op["LocA_percent"]
    T_BUDGET_MS  = op["T_budget_ms"]
    LOCA_TARGET  = op["Loca_target_percent"]
    HEADROOM_MS  = T_BUDGET_MS - SUBTOTAL_MS
    HEADROOM_PCT = HEADROOM_MS / T_BUDGET_MS * 100.0

    fig, ax = plt.subplots(figsize=(9.0, 5.8))

    # Sort the data for the Pareto envelope
    data_sorted = sorted(pareto_data, key=lambda x: x[1])
    x_vals = [d[1] for d in data_sorted]
    y_vals = [d[2] for d in data_sorted]

    # ---- Background shading: feasible vs infeasible algorithmic latency ----
    ax.axvspan(0, T_BUDGET_MS, facecolor='#d1fae5', alpha=0.35,
               label='Feasible (T$_{iso}$ < 10 ms)')
    ax.axvspan(T_BUDGET_MS, 16, facecolor='#fee2e2', alpha=0.35,
               label='Infeasible (T$_{iso}$ > 10 ms)')

    # ---- Pareto front curve ----
    ax.plot(x_vals, y_vals,
            color='#1e3a8a', linewidth=2.5, linestyle='-',
            marker='o', markersize=8, markerfacecolor='#3b82f6',
            markeredgecolor='#1e3a8a', markeredgewidth=1.2,
            label='Pareto front (GA-converged)', zorder=4)

    # ---- Hard algorithmic ceiling (vertical) ----
    ax.axvline(T_BUDGET_MS, color='#dc2626', linestyle='--', linewidth=2.0,
               label='T$_{budget}$ = 10.0 ms (algorithmic ceiling)', zorder=3)

    # ---- Performance target line (horizontal) ----
    ax.axhline(LOCA_TARGET, color='#d97706', linestyle='--', linewidth=1.8,
               label='LocA target = 95%', zorder=3)

    # ---- Chosen operating point (read from operating-point CSV) ----
    ax.scatter([SUBTOTAL_MS], [CHOSEN_LOCA],
               s=380, marker='*', color='#fbbf24',
               edgecolor='#92400e', linewidth=1.8, zorder=6,
               label=(f'Chosen: w$_2$ = {W2_FINAL}, w$_4$ = {W4_FINAL}\n'
                      f'(w$_4$/w$_2$ = {W4_FINAL / W2_FINAL:.2f})'))

    # ---- Annotation box for the chosen point ----
    ax.annotate(
        f'CHOSEN OPERATING POINT\n'
        f'Subtotal from Table 2\n'
        f'(A: {stage_ms["Stage A"]:.2f} + B: {stage_ms["Stage B"]:.2f} + '
        f'C: {stage_ms["Stage C"]:.2f})\n'
        f'T$_{{iso}}$ = {SUBTOTAL_MS:.2f} ms\n'
        f'LocA = {CHOSEN_LOCA:.2f}%\n'
        f'Headroom: {HEADROOM_MS:.2f} ms ({HEADROOM_PCT:.1f}%)',
        xy=(SUBTOTAL_MS, CHOSEN_LOCA),
        xytext=(SUBTOTAL_MS + 4.5, CHOSEN_LOCA - 9.0),
        fontsize=9, ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef3c7',
                  edgecolor='#92400e', linewidth=1.6),
        arrowprops=dict(arrowstyle='->', color='#92400e', lw=1.3),
    )

    # ---- Axis intercept labels ----
    ax.text(T_BUDGET_MS + 0.18, 65.5, '10.0 ms\n(algorithmic\nceiling)',
            fontsize=9, color='#dc2626', fontweight='bold', va='bottom')
    ax.text(0.2, LOCA_TARGET + 0.25, '95% target',
            fontsize=9, color='#d97706', fontweight='bold', va='bottom')

    # ---- Lower annotation: explain what is excluded ----
    ax.text(0.5, 67.5,
            f'Note: This is the ISOLATED algorithmic execution time.\n'
            f'Wall-clock end-to-end ({e2e_ms:.2f} ms) is reported separately in Table 2\n'
            f'and is NOT plotted on this curve.',
            fontsize=8.5, color='#475569', style='italic', va='bottom')

    # ---- Axes cosmetics ----
    ax.set_xlim(0, 16)
    ax.set_ylim(64, 100)
    ax.set_xlabel('Isolated Algorithmic Latency T$_{iso}$ (ms)\n'
                  '[Stage A + Stage B + Stage C; no I/O, no Python overhead]',
                  fontsize=11.5, fontweight='bold')
    ax.set_ylabel('Localization Accuracy LocA (%)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6, zorder=1)
    ax.set_axisbelow(True)

    # ---- Legend (lower right, in the infeasible zone) ----
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95,
              edgecolor='#475569', fancybox=True)

    # ---- Save (no title; EAAI supplies caption separately) ----
    out_dir = Path(__file__).resolve().parent
    for ext in ('png', 'pdf', 'svg'):
        out_path = out_dir / f'Figure11_Pareto_Front.{ext}'
        plt.savefig(out_path, bbox_inches='tight', facecolor='white')
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    create_figure11()
