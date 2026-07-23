"""
Table 3: Isolated State-Space Estimator Latency Benchmark.

Reports the isolated algorithmic latency of the three online tracking
back-ends (STMSS linear KF, IMM, and 500-particle PF). The data
lives in ``data/csv/table3_state_space.csv`` as the single source of
truth. This script does NOT re-measure the latencies; it reads the
canonical values from the CSV, validates internal consistency, and
prints a human-readable summary.

Validation rules (EAAI consistency):
    1. If Mean > 20 ms, Violation_Rate_Percent must be non-trivial (> 1%).
       A live re-benchmark that yields Mean > 20 ms with 0% violations
       indicates that the measurement was performed with a different
       cycle budget than the rest of Table 3.
    2. The Complexity column must be one of {O(1), O(K^2), O(N log N)}.

Outputs:
    - Prints the table to stdout.
    - Exits with code 0 on success, 1 on inconsistency.
"""

import csv
import sys
from pathlib import Path

CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "csv"
    / "table3_state_space.csv"
)

ALLOWED_COMPLEXITY = {"O(1)", "O(K^2)", "O(N log N)"}


def load_canonical_table(csv_path: Path) -> list:
    """Load Table 3 from the canonical CSV."""
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
    bar = "-" * 96
    head = (f"{'Estimator':<40} {'Mean (ms)':>10} {'P95 (ms)':>10} "
            f"{'Violations':>12} {'Rate (%)':>10} {'Complexity':>14}")
    out = [bar, head, bar]
    for r in rows:
        out.append(
            f"{r['Estimator'][:40]:<40} "
            f"{_f(r, 'Mean_ms'):>10.3f} "
            f"{_f(r, 'P95_ms'):>10.3f} "
            f"{int(_f(r, 'Violations_20ms')):>12d} "
            f"{_f(r, 'Violation_Rate_Percent'):>10.1f} "
            f"{r.get('Complexity', ''):>14}"
        )
    out.append(bar)
    return "\n".join(out)


def validate(rows: list) -> list:
    """Ensure the table is internally consistent. Return a list of errors."""
    errors = []
    for r in rows:
        est = r["Estimator"].strip()
        mean = _f(r, "Mean_ms")
        rate = _f(r, "Violation_Rate_Percent")
        complexity = r.get("Complexity", "").strip()
        # EAAI consistency rule: if mean > 20 ms, violation rate must
        # be non-trivial (> 1%).
        if mean > 20.0 and rate < 1.0:
            errors.append(
                f"{est}: mean={mean:.3f}ms > 20ms but violation rate={rate:.1f}%. "
                f"This violates the Table 3 EAAI consistency rule "
                f"(mean > 20 ms requires non-trivial violation rate)."
            )
        # Complexity must be one of the three allowed labels.
        if complexity and complexity not in ALLOWED_COMPLEXITY:
            errors.append(
                f"{est}: complexity '{complexity}' is not one of "
                f"{sorted(ALLOWED_COMPLEXITY)}."
            )
    return errors


def main() -> int:
    rows = load_canonical_table(CSV_PATH)
    print("=" * 96)
    print("Table 3: Isolated State-Space Estimator Latency")
    print("=" * 96)
    print(render(rows))
    print(f"CSV source: {CSV_PATH}")
    print()
    errors = validate(rows)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("Validation OK: EAAI consistency rules satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
