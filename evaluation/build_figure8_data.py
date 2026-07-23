"""
Figure 8 data builder: produces ``data/csv/figure8_survival_probability.csv``
from the two raw per-sequence baseline CSVs plus a deterministic
escape-boundary curve computed from the physical model.

This script is the SINGLE source of truth for the per-pipeline T_comp /
P_capture points used by ``figures/plot_figure8_survival_probability.py``.

The values flow directly from the raw Culex_Transit measurements and the
deterministic escape-boundary formula; no manuscript-locked, hand-edited
or hard-coded values are used anywhere in this script.

Data flow:

    data/csv/table1_semantic_baselines.csv
        YOLOv8n+ByteTrack, STMSS, eight sequences each. The
        Culex_Transit row supplies the T_comp for those two
        pipelines in figure 8.

    data/csv/table1_baselines.csv
        MOG2 + Lucas-Kanade, MOG2 + SORT, MOG2 + IMM (2-Model),
        eight sequences each. The Culex_Transit row supplies the
        T_comp for those three pipelines in figure 8.

Physical model (curve rows):

    T_allow(T_comp) = exp(-T_comp / tau) * 100%
    P_capture(T_comp) = clamp(T_allow_curve(T_comp), 0, 100)

where the exponential time-constant tau is fit to the physical model
R / v_esc = 15 cm / 2.0 m/s = 75 ms (the deterministic worst-case
allowance) and a half-saturation empirical anchor at T_comp = 75 ms
yields P_capture = 50%. The curve rows sample this deterministic
function at 5 ms intervals; they are NOT manuscript-locked.

The pipeline rows are the raw Culex_Transit T_comp measurements for the
five pipelines; their P_capture values are read off the deterministic
curve at that T_comp (i.e. computed, not measured).

Output:
    data/csv/figure8_survival_probability.csv
"""

import csv
import math
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data" / "csv"

# Physical constants (NOT data)
R_M = 0.15                  # 15 cm effective capture radius
V_ESC = 2.0                 # 2 m/s critical sprint velocity
T_DEADLINE_MS = R_M / V_ESC * 1000.0   # = 75.0 ms (strict radial)

# Curve-sampling grid
CURVE_T_MIN = 5.0
CURVE_T_MAX = 75.0
CURVE_T_STEP = 5.0

# Pipeline order (presentation only)
ORDER = [
    "YOLOv8n + ByteTrack",
    "MOG2 + Lucas-Kanade",
    "MOG2 + SORT",
    "MOG2 + IMM (2-Model)",
    "STMSS (Proposed)",
]

INTERNAL_KEY = {
    "YOLOv8n + ByteTrack":  "YOLOv8n_ByteTrack",
    "MOG2 + Lucas-Kanade":  "MOG2_LK",
    "MOG2 + SORT":          "MOG2_SORT",
    "MOG2 + IMM (2-Model)": "MOG2_IMM",
    "STMSS (Proposed)":     "STMSS",
}

CANONICAL_SEQUENCE = "Culex_Transit"


def _load_csv_dicts(path: Path) -> list:
    """Read a CSV. Lines starting with '#' (after optional whitespace)
    are skipped, so the data files can carry a provenance header.
    """
    rows = []
    header = None
    with open(path, "r", newline="") as fh:
        for line in fh:
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            cells = next(csv.reader([line.rstrip("\r\n")]))
            if header is None:
                header = cells
                continue
            rows.append(dict(zip(header, cells)))
    return rows


def _p_capture_curve(t_comp_ms: float) -> float:
    """Deterministic P_capture(T_comp) curve. Monotonically decreasing
    from 100% at T_comp = 0 to 50% at T_comp = 75.0 ms (the
    aerodynamic deadline). Empirically anchored; not a measurement.
    """
    if t_comp_ms <= 0:
        return 100.0
    # Exponential decay with tau = 75 / ln(2) ms so that
    # P(75 ms) = 50% (half-saturation at the deadline).
    tau = T_DEADLINE_MS / math.log(2.0)
    return max(0.0, min(100.0, 100.0 * math.exp(-t_comp_ms / tau)))


def _read_canonical_tcomp() -> dict:
    """Return {display_name: T_comp_ms} for the Culex_Transit row of
    each of the five pipelines, read directly from the raw baseline
    CSVs.
    """
    semantic = {}
    for row in _load_csv_dicts(DATA_DIR / "table1_semantic_baselines.csv"):
        semantic.setdefault(row["Pipeline"], {})[row["Sequence"]] = \
            float(row["Mean_ms"])

    mog2 = {}
    for row in _load_csv_dicts(DATA_DIR / "table1_baselines.csv"):
        mog2.setdefault(row["Pipeline"], {})[row["Sequence"]] = \
            float(row["Mean_ms"])

    out = {}
    for display in ORDER:
        key = INTERNAL_KEY[display]
        per_seq = semantic.get(key) or mog2.get(key)
        if per_seq is None:
            print(f"[warn] no raw measurements for {display}, skipping")
            continue
        if CANONICAL_SEQUENCE not in per_seq:
            print(f"[warn] {display}: no {CANONICAL_SEQUENCE} row, skipping")
            continue
        out[display] = per_seq[CANONICAL_SEQUENCE]
    return out


def build_figure8_rows() -> list:
    """Build the curve rows + pipeline rows for figure 8."""
    rows = []
    # 1. Curve rows: deterministic P_capture bound sampled on a grid
    t = CURVE_T_MIN
    while t <= CURVE_T_MAX + 0.001:
        rows.append({
            "T_comp_ms": f"{t:.2f}",
            "P_capture_percent": f"{_p_capture_curve(t):.2f}",
            "Pipeline": "Curve",
        })
        t += CURVE_T_STEP

    # 2. Pipeline rows: T_comp is the raw Culex_Transit mean latency
    tcomp = _read_canonical_tcomp()
    for display in ORDER:
        if display not in tcomp:
            continue
        t_comp = tcomp[display]
        rows.append({
            "T_comp_ms": f"{t_comp:.2f}",
            "P_capture_percent": f"{_p_capture_curve(t_comp):.2f}",
            "Pipeline": display,
        })
    return rows


def write_csv(rows: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T_comp_ms", "P_capture_percent", "Pipeline"])
        for r in rows:
            w.writerow([r["T_comp_ms"], r["P_capture_percent"], r["Pipeline"]])
    print(f"Wrote {out_path}")


def main():
    rows = build_figure8_rows()
    if not rows:
        raise SystemExit("No figure-8 rows; check raw baseline CSVs.")
    write_csv(rows, DATA_DIR / "figure8_survival_probability.csv")


if __name__ == "__main__":
    main()
