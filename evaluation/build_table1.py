"""
Table 1 generator for the manuscript.

Reads the *raw* per-sequence latency measurements from the
``data/csv/table1_*.csv`` files and emits the aggregated Table 1
output (``data/csv/table1.csv`` plus a markdown rendering at
``docs/Table1.md``).

All quantitative data flows from the two raw measurement CSVs:

    data/csv/table1_semantic_baselines.csv
        YOLOv8n+ByteTrack and STMSS, eight sequences each.
        Source: the actual benchmark runs of those two pipelines
        on the corresponding test sequences. The Culex_Transit
        row is the deadline-calibration reference (mean ± std of
        the per-frame latency over the sequence's frames).
    data/csv/table1_baselines.csv
        MOG2 + Lucas-Kanade, MOG2 + SORT, MOG2 + IMM (2-Model),
        eight sequences each. Same source.

    data/csv/table1_metadata.csv
        Pipeline -> hardware profile (configuration), and the
        debris rejection rate per pipeline on the
        Wind_Debris_Augmented sequence. The hardware label is a
        configuration; the debris rejection rate is the result of
        running each pipeline on the Wind_Debris_Augmented
        evaluation sequence. The raw per-sequence debris rates
        for the MOG2 pipelines are not split out into a per-
        sequence CSV; the single Wind_Debris_Aug value is the
        aggregate over the sequence and is what the manuscript
        quotes.

The mechanical solenoid delay (25.0 ms) and the aerodynamic
deadline (75.0 ms) are the only physical constants in the script;
they come from the EXAIR Super Air Knife manufacturer datasheet
and the 2.0 m/s / 15 cm / 0.5-frame penetration model
respectively, not from any CSV.

Outputs:
    data/csv/table1.csv
    docs/Table1.md
"""

import csv
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data" / "csv"
TABLES_DIR = THIS_DIR.parent / "docs"

# Physical constants (manufacturer datasheet + physical model, NOT data)
MECHANICAL_DELAY_MS = 25.0
DEADLINE_MS = 75.0

# Display names (presentation only, not data)
DISPLAY = {
    "YOLOv8n_ByteTrack": "YOLOv8n + ByteTrack",
    "MOG2_LK":           "MOG2 + Lucas-Kanade",
    "MOG2_SORT":         "MOG2 + SORT",
    "MOG2_IMM":          "MOG2 + IMM (2-Model)",
    "STMSS":             "STMSS (Proposed)",
}


def _load_csv_dicts(path: Path) -> list:
    """Read a CSV file. Lines starting with '#' (after optional leading
    whitespace) are treated as comments and skipped, so the data files
    can carry provenance notes as a header block. The first
    non-comment line is the header row."""
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


def _load_csv_sections(path: Path) -> dict:
    """Read a multi-section CSV. Section header is a bracketed line
    like ``[Section_Name]`` followed by a header row. Returns
    ``{"Section_Name": [rows...]}``.

    Lines starting with ``#`` (after optional leading whitespace) are
    treated as comments and skipped.
    """
    sections = {}
    current_name = None
    current_header = None
    with open(path, "r", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            first = row[0].lstrip()
            if first.startswith("#"):
                continue
            if first.startswith("[") and first.endswith("]"):
                current_name = first.strip("[]")
                current_header = next(reader, None)
                if current_header is None:
                    continue
                sections[current_name] = []
                continue
            if current_name is None:
                continue
            sections[current_name].append(dict(zip(current_header, row)))
    return sections


def load_semantic_baselines() -> dict:
    """Return {pipeline_name: {sequence: {mean, std, p95}}} for the
    two semantic pipelines (YOLOv8n+ByteTrack, STMSS). These are the
    raw per-frame latency statistics, one row per sequence."""
    table = {}
    for row in _load_csv_dicts(DATA_DIR / "table1_semantic_baselines.csv"):
        table.setdefault(row["Pipeline"], {})[row["Sequence"]] = {
            "mean": float(row["Mean_ms"]),
            "std":  float(row["Std_ms"]),
            "p95":  float(row["P95_ms"]),
        }
    return table


def load_mog2_baselines() -> dict:
    """Return {pipeline_name: {sequence: {mean, std, p95}}} for the
    three MOG2 baselines."""
    table = {}
    for row in _load_csv_dicts(DATA_DIR / "table1_baselines.csv"):
        table.setdefault(row["Pipeline"], {})[row["Sequence"]] = {
            "mean": float(row["Mean_ms"]),
            "std":  float(row["Std_ms"]),
            "p95":  float(row["P95_ms"]),
        }
    return table


def load_metadata() -> tuple:
    """Return (hardware_dict, debris_dict) from the metadata CSV."""
    sections = _load_csv_sections(DATA_DIR / "table1_metadata.csv")
    hardware = {r["Pipeline"]: r["Hardware"]
                for r in sections.get("Pipeline", [])}
    debris = {r["Pipeline"]: float(r["Rate"])
              for r in sections.get("Debris_Rejection_Percent", [])}
    return hardware, debris


def status(total):
    if total > DEADLINE_MS:
        return "FAIL"
    if total > 60.0:
        return "MARGINAL"
    return "GUARANTEED"


def build_table():
    """Aggregate the per-sequence raw measurements into a single
    Table 1 view, anchored on the Culex_Transit deadline-calibration
    reference row. The values here come directly from the two raw CSVs.

    The "Culex_Transit reference" means: the row's mean, std and
    P95 are the per-frame statistics of that sequence. The other
    seven sequences (Aedes_Saccade, Drosophila_Dense,
    LevyTest_alpha05/10/15, Synthetic_Swarm, Wind_Debris_Aug) are
    also present in the same CSV but are not aggregated into the
    headline mean -- the manuscript anchors the deadline on a
    single sequence, and the Culex_Transit row is that anchor.
    """
    semantic = load_semantic_baselines()
    mog2 = load_mog2_baselines()
    hardware, debris = load_metadata()

    CANONICAL = "Culex_Transit"

    sequences = list(semantic["STMSS"].keys())

    aggregated = {}
    pipeline_keys = {
        "YOLOv8n + ByteTrack":  "YOLOv8n_ByteTrack",
        "MOG2 + Lucas-Kanade":  "MOG2_LK",
        "MOG2 + SORT":          "MOG2_SORT",
        "MOG2 + IMM (2-Model)": "MOG2_IMM",
        "STMSS (Proposed)":     "STMSS",
    }
    for display_name, internal_key in pipeline_keys.items():
        if internal_key in semantic:
            per_seq = semantic[internal_key]
        elif internal_key in mog2:
            per_seq = mog2[internal_key]
        else:
            print(f"[warn] no raw measurements for {display_name}, skipping")
            continue

        if CANONICAL not in per_seq:
            print(f"[warn] {display_name}: no Culex_Transit row, skipping")
            continue

        ref = per_seq[CANONICAL]
        aggregated[display_name] = {
            "per_seq": per_seq,
            "mean": ref["mean"],
            "std":  ref["std"],
            "p95":  ref["p95"],
            "hardware": hardware.get(display_name, "Pure CPU"),
            "debris_rejection": debris.get(display_name),
        }

    return aggregated, sequences


def write_csv(aggregated, sequences, out_path):
    with open(out_path, "w", newline="") as f:
        f.write(
            "Pipeline,Hardware,Inference_Mean_ms,Inference_Std_ms,Inference_P95_ms,"
            "Mechanical_ms,Total_ms,Deadline_ms,Margin_ms,Status,"
            "Debris_Rejection_Percent\n"
        )
        for name, agg in aggregated.items():
            total = agg["mean"] + MECHANICAL_DELAY_MS
            margin = DEADLINE_MS - total
            rej = agg["debris_rejection"]
            rej_str = f"{rej:.1f}" if rej is not None else "NA"
            f.write(
                f"{name},{agg['hardware']},"
                f"{agg['mean']:.3f},{agg['std']:.3f},"
                f"{agg['p95']:.3f},{MECHANICAL_DELAY_MS:.1f},{total:.3f},"
                f"{DEADLINE_MS:.1f},{margin:.3f},{status(total)},{rej_str}\n"
            )
    print(f"Wrote {out_path}")


def write_markdown(aggregated, sequences, out_path):
    lines = []
    lines.append("# Table 1. End-to-End Cyber-Physical Latency and Tracking Benchmarks")
    lines.append("")
    lines.append("Hardware: Intel Core i5-8250U @ 1.6 GHz, 8 GB RAM (CPU only). "
                 "All measurements on the same 512x512 down-sampled stream.")
    lines.append("")
    lines.append(
        "**What this table reports.** Each row is the *Culex_Transit* "
        "deadline-calibration reference row from the raw per-sequence "
        "measurement CSVs (`data/csv/table1_semantic_baselines.csv` and "
        "`data/csv/table1_baselines.csv`). The other seven sequences are "
        "present in those same CSVs for audit but are not aggregated into "
        "the headline mean. The headline mean, σ and P95 are the per-frame "
        "statistics of the Culex_Transit sequence under the EAAI latency "
        "protocol (pre-loaded observations, `gc.disable()`, 100-iteration "
        "cache warm-up, locked CPU frequency). The values flow "
        "directly from the raw CSVs into the rendered output."
    )
    lines.append("")
    lines.append(
        "| Pipeline | Hardware | T_comp (ms) | σ (ms) | P95 (ms) "
        "| T_mech (ms) | T_total (ms) | T_allow (ms) | Margin (ms) "
        "| Debris Rejection | Actuation Status |"
    )
    lines.append(
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    )
    for name, agg in aggregated.items():
        total = agg["mean"] + MECHANICAL_DELAY_MS
        margin = DEADLINE_MS - total
        rej = agg["debris_rejection"]
        rej_str = f"{rej:.1f}%" if rej is not None else "N/A (Semantic)"
        lines.append(
            f"| {name} | {agg['hardware']} "
            f"| {agg['mean']:.2f} | {agg['std']:.2f} "
            f"| {agg['p95']:.2f} "
            f"| {MECHANICAL_DELAY_MS:.1f} | {total:.2f} "
            f"| {DEADLINE_MS:.1f} | {margin:.2f} | {rej_str} "
            f"| {status(total)} |"
        )
    lines.append("")
    lines.append("Notes: T_allow = 75.0 ms is the deterministic physical upper "
                 "bound for a 2.0 m/s orthogonal micro-vector penetrating the "
                 "15 cm effective capture radius of an industrial air curtain. "
                 "T_mech = 25.0 ms is the industrial solenoid activation lag "
                 "(hardware-certified worst-case per the EXAIR Super Air Knife "
                 "manufacturer datasheet). Debris Rejection Rate is evaluated on "
                 "the Wind_Debris_Augmented sequence. The P99 worst-case "
                 "execution time is not stored in this repo; the figure-7 "
                 "waterfall reads directly from the raw Culex_Transit CSVs.")
    lines.append("")
    lines.append("Script: `evaluation/build_table1.py`")
    lines.append("Data:")
    lines.append("- `data/csv/table1_semantic_baselines.csv` (YOLOv8n, STMSS per-sequence)")
    lines.append("- `data/csv/table1_baselines.csv` (MOG2 raw per-sequence measurements)")
    lines.append("- `data/csv/table1_metadata.csv` (hardware + debris rejection)")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def main():
    aggregated, sequences = build_table()
    if not aggregated:
        raise SystemExit(
            "No baseline measurements found. Check "
            "data/csv/table1_baselines.csv and "
            "data/csv/table1_semantic_baselines.csv."
        )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(aggregated, sequences, DATA_DIR / "table1.csv")
    write_markdown(aggregated, sequences, TABLES_DIR / "Table1.md")


if __name__ == "__main__":
    main()
