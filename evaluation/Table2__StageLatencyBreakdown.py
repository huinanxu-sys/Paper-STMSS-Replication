"""
Table 2: Stage-by-Stage Latency Breakdown.

Reports the per-stage computational cost of the STMSS pipeline. The
data lives in ``data/csv/table2_stage_latency.csv`` as the single
source of truth. This script does NOT re-measure the latencies; it
reads the canonical values from the CSV, validates internal
consistency, and prints a human-readable summary.

The values in the CSV are the integrated-pipeline measurements
performed under the EAAI isolation protocol on the reference
i5-8250U platform. The ``Subtotal`` row is the arithmetic sum of
Stages A, B, and C (i.e. pure compute, no I/O); the ``Overhead`` row
is the difference between the integrated end-to-end wall-clock and
the Subtotal, representing the cost of video I/O, decoding, memory
management and the Python toolchain.

Output:
    - Prints the table to stdout.
    - Validates Subtotal == A + B + C and End-to-End == Subtotal + Overhead.
    - Exits with code 0 on success, 1 on inconsistency.
"""

import csv
import sys
from pathlib import Path

CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "csv"
    / "table2_stage_latency.csv"
)


def load_canonical_table(csv_path: Path) -> list:
    """Load Table 2 from the canonical CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Canonical CSV not found: {csv_path}")
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _f(row, key, default=0.0):
    raw = row.get(key, "").strip()
    if raw == "":
        return default
    return float(raw)


def render(rows: list) -> str:
    """Return a printable ASCII version of the table."""
    bar = "-" * 92
    head = f"{'Component':<14} {'Description':<55} {'Mean (ms)':>10} {'% of Total':>10}"
    out = [bar, head, bar]
    for r in rows:
        out.append(
            f"{r['Component']:<14} "
            f"{r['Description'][:55]:<55} "
            f"{_f(r, 'Mean_ms'):>10.2f} "
            f"{_f(r, 'Percentage_of_Total'):>9.1f}%"
        )
    out.append(bar)
    return "\n".join(out)


def validate(rows: list) -> list:
    """Ensure the table is internally consistent. Return a list of errors."""
    errors = []
    by_name = {r["Component"]: r for r in rows}
    a = _f(by_name["Stage A"], "Mean_ms")
    b = _f(by_name["Stage B"], "Mean_ms")
    c = _f(by_name["Stage C"], "Mean_ms")
    sub = _f(by_name["Subtotal"], "Mean_ms")
    ovh = _f(by_name["Overhead"], "Mean_ms")
    e2e = _f(by_name["End-to-End"], "Mean_ms")

    # The arithmetic must be self-consistent.
    if abs(sub - (a + b + c)) > 0.01:
        errors.append(
            f"Subtotal ({sub:.3f}) != Stage A ({a:.3f}) + B ({b:.3f}) + C ({c:.3f})"
        )
    if abs(e2e - (sub + ovh)) > 0.01:
        errors.append(
            f"End-to-End ({e2e:.3f}) != Subtotal ({sub:.3f}) + Overhead ({ovh:.3f})"
        )
    if abs(_f(by_name["End-to-End"], "Percentage_of_Total") - 100.0) > 0.01:
        errors.append("End-to-End row must report 100.0% of total")
    return errors


def main() -> int:
    rows = load_canonical_table(CSV_PATH)
    print("=" * 92)
    print("Table 2: Stage-by-Stage Latency Breakdown")
    print("=" * 92)
    print(render(rows))
    print(f"CSV source: {CSV_PATH}")
    print()
    errors = validate(rows)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("Validation OK: Subtotal = A + B + C; End-to-End = Subtotal + Overhead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
