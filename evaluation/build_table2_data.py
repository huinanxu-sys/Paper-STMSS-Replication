"""
Table 2 builder: produces ``data/csv/table2_stage_latency.csv`` from
two raw per-stage CSVs and the raw Table 1 measurement CSV.

This script is the SINGLE source of truth for Table 2. The values flow
directly from the raw measurements.

Data flow:

    data/csv/table2_stage_latency_measurements.csv
        Per-stage latency (Stage A, Stage B, Stage C) under the EAAI
        isolation protocol. Source: the actual benchmark runs of each
        isolated stage on the reference Culex_Transit stream.

    data/csv/table1_semantic_baselines.csv
        STMSS Culex_Transit mean latency. Source: the integrated
        end-to-end benchmark under the same EAAI protocol. Used to
        derive the End-to-End and Overhead rows.

Derived rows (in the rendered CSV):
    Subtotal  = Stage A + Stage B + Stage C  (pure compute)
    Overhead  = End-to-End - Subtotal         (I/O + toolchain)
    End-to-End = STMSS Culex_Transit mean   (read from Table 1 raw)

Output:
    data/csv/table2_stage_latency.csv
"""

import csv
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data" / "csv"

STAGE_CSV      = DATA_DIR / "table2_stage_latency_measurements.csv"
STMSS_RAW_CSV  = DATA_DIR / "table1_semantic_baselines.csv"
RENDERED_CSV   = DATA_DIR / "table2_stage_latency.csv"

CANONICAL_SEQUENCE = "Culex_Transit"
STMSS_KEY           = "STMSS"


def _load_csv_dicts(path: Path) -> list:
    """Read a CSV. Lines starting with '#' (after optional whitespace)
    are skipped, so the data files can carry a provenance header."""
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


def _read_stmss_e2e_ms() -> float:
    """Return the STMSS Culex_Transit mean latency from the raw
    per-frame Table 1 measurement CSV."""
    for row in _load_csv_dicts(STMSS_RAW_CSV):
        if (row.get("Pipeline") == STMSS_KEY
                and row.get("Sequence") == CANONICAL_SEQUENCE):
            return float(row["Mean_ms"])
    raise RuntimeError(
        f"STMSS Culex_Transit row not found in {STMSS_RAW_CSV}"
    )


def _read_stage_measurements() -> dict:
    """Return {Component: row_dict} for the three stages from the
    canonical per-stage measurement CSV. The dict includes
    Description, Mean_ms and Measurement_Protocol, all of which
    are propagated to the rendered output so this script does not
    hard-code any protocol string."""
    out = {}
    for row in _load_csv_dicts(STAGE_CSV):
        comp = row["Component"].strip()
        out[comp] = row
    for required in ("Stage A", "Stage B", "Stage C"):
        if required not in out:
            raise RuntimeError(
                f"{required} row missing from {STAGE_CSV}"
            )
    return out


def build_table2_rows(stage_rows: dict, e2e_ms: float) -> list:
    """Return the rendered Table 2 row list. The Subtotal, Overhead
    and End-to-End rows are DERIVED (not stored) so the rendered CSV
    cannot drift away from the raw measurements."""
    a = float(stage_rows["Stage A"]["Mean_ms"])
    b = float(stage_rows["Stage B"]["Mean_ms"])
    c = float(stage_rows["Stage C"]["Mean_ms"])
    subtotal = a + b + c
    overhead = e2e_ms - subtotal

    def stage_row(name):
        r = stage_rows[name]
        mean = float(r["Mean_ms"])
        return {
            "Component":            r["Component"],
            "Description":           r["Description"],
            "Mean_ms":              f"{mean:.2f}",
            "Percentage_of_Total":  f"{mean / e2e_ms * 100.0:.1f}",
            "Measurement_Protocol": r["Measurement_Protocol"],
        }

    return [
        stage_row("Stage A"),
        stage_row("Stage B"),
        stage_row("Stage C"),
        {
            "Component":            "Subtotal",
            "Description":           "Sum of Stages A + B + C (pure compute)",
            "Mean_ms":              f"{subtotal:.2f}",
            "Percentage_of_Total":  f"{subtotal / e2e_ms * 100.0:.1f}",
            "Measurement_Protocol": "Arithmetic sum of stages A + B + C",
        },
        {
            "Component":            "Overhead",
            "Description":           "Video I/O, Decoding, Memory Management, Python Toolchain",
            "Mean_ms":              f"{overhead:.2f}",
            "Percentage_of_Total":  f"{overhead / e2e_ms * 100.0:.1f}",
            "Measurement_Protocol": "Derived: End-to-End minus Subtotal",
        },
        {
            "Component":            "End-to-End",
            "Description":           "Full pipeline wall-clock (STMSS Culex_Transit reference)",
            "Mean_ms":              f"{e2e_ms:.2f}",
            "Percentage_of_Total":  "100.0",
            "Measurement_Protocol": "EAAI isolation, gc disabled, 100 warmup iters",
        },
    ]


def write_csv(rows: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["Component", "Description", "Mean_ms",
               "Percentage_of_Total", "Measurement_Protocol"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_path}")


def main():
    stage_rows = _read_stage_measurements()
    e2e_ms = _read_stmss_e2e_ms()
    rows = build_table2_rows(stage_rows, e2e_ms)
    write_csv(rows, RENDERED_CSV)


if __name__ == "__main__":
    main()
