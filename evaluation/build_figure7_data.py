"""
Figure 7 data builder: produces ``data/csv/figure7_latency_data.csv`` from
the two raw per-sequence baseline CSVs.

This script is the SINGLE source of truth for the per-pipeline latency
rows used by ``figures/plot_figure7_latency_waterfall.py``. The values
flow directly from the raw Culex_Transit measurements -- no manuscript-
locked, hand-edited or hard-coded values are used anywhere in this
script.

Data flow:

    data/csv/table1_semantic_baselines.csv
        YOLOv8n+ByteTrack, STMSS, eight sequences each. The
        Culex_Transit row is the deadline-calibration reference.

    data/csv/table1_baselines.csv
        MOG2 + Lucas-Kanade, MOG2 + SORT, MOG2 + IMM (2-Model),
        eight sequences each. The Culex_Transit row is the
        deadline-calibration reference.

    data/csv/table1_metadata.csv
        Pipeline -> hardware profile (configuration only).

The mechanical solenoid delay (25.0 ms) and the aerodynamic deadline
(75.0 ms) are the only physical constants; they come from the
EXAIR Super Air Knife manufacturer datasheet and the 2.0 m/s /
15 cm / 0.5-frame penetration model, not from any CSV.

Output:
    data/csv/figure7_latency_data.csv
"""

import csv
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data" / "csv"

# Physical constants (NOT data, see module docstring)
MECHANICAL_DELAY_MS = 25.0
DEADLINE_MS = 75.0

# Display order (presentation only, not data)
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


def _load_metadata(path: Path) -> dict:
    """Return {display_name: hardware} from the [Pipeline] section."""
    sections = {}
    current = None
    cur_header = None
    with open(path, "r", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            first = row[0].lstrip()
            if first.startswith("#"):
                continue
            if first.startswith("[") and first.endswith("]"):
                current = first.strip("[]")
                cur_header = next(reader, None)
                if cur_header is None:
                    continue
                sections[current] = []
                continue
            if current is None:
                continue
            sections[current].append(dict(zip(cur_header, row)))
    return {r["Pipeline"]: r["Hardware"]
            for r in sections.get("Pipeline", [])}


def _status(total: float) -> str:
    if total > DEADLINE_MS:
        return "FAIL"
    if total > 60.0:
        return "MARGINAL"
    return "GUARANTEED"


def build_figure7_rows() -> list:
    """Read the raw CSVs and assemble the five pipeline rows for
    figure 7 (Culex_Transit reference row of each pipeline).
    """
    semantic = {}
    for row in _load_csv_dicts(DATA_DIR / "table1_semantic_baselines.csv"):
        semantic.setdefault(row["Pipeline"], {})[row["Sequence"]] = {
            "mean": float(row["Mean_ms"]),
            "std":  float(row["Std_ms"]),
            "p95":  float(row["P95_ms"]),
        }

    mog2 = {}
    for row in _load_csv_dicts(DATA_DIR / "table1_baselines.csv"):
        mog2.setdefault(row["Pipeline"], {})[row["Sequence"]] = {
            "mean": float(row["Mean_ms"]),
            "std":  float(row["Std_ms"]),
            "p95":  float(row["P95_ms"]),
        }

    hardware = _load_metadata(DATA_DIR / "table1_metadata.csv")

    rows = []
    for display in ORDER:
        key = INTERNAL_KEY[display]
        per_seq = semantic.get(key) or mog2.get(key)
        if per_seq is None:
            print(f"[warn] no raw measurements for {display}, skipping")
            continue
        ref = per_seq.get(CANONICAL_SEQUENCE)
        if ref is None:
            print(f"[warn] {display}: no {CANONICAL_SEQUENCE} row, skipping")
            continue
        infer = ref["mean"]
        total = infer + MECHANICAL_DELAY_MS
        rows.append({
            "System": display,
            "Inference_Latency_ms": f"{infer:.3f}",
            "Mechanical_Latency_ms": f"{MECHANICAL_DELAY_MS:.1f}",
            "Total_Latency_ms": f"{total:.3f}",
            "Deadline_ms": f"{DEADLINE_MS:.1f}",
            "Margin_ms": f"{DEADLINE_MS - total:.3f}",
            "Status": _status(total),
            "Hardware": hardware.get(display, "Pure CPU"),
        })
    return rows


def write_csv(rows: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "System",
            "Inference_Latency_ms",
            "Mechanical_Latency_ms",
            "Total_Latency_ms",
            "Deadline_ms",
            "Margin_ms",
            "Status",
            "Hardware",
        ])
        for r in rows:
            w.writerow([
                r["System"],
                r["Inference_Latency_ms"],
                r["Mechanical_Latency_ms"],
                r["Total_Latency_ms"],
                r["Deadline_ms"],
                r["Margin_ms"],
                r["Status"],
                r["Hardware"],
            ])
    print(f"Wrote {out_path}")


def main():
    rows = build_figure7_rows()
    if not rows:
        raise SystemExit("No raw baseline rows; cannot build figure7 data.")
    write_csv(rows, DATA_DIR / "figure7_latency_data.csv")


if __name__ == "__main__":
    main()
