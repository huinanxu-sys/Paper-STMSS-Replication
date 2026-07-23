"""
Table 5: Ablation of the Size-Velocity Temporal Variance Filter
on Non-Biological Debris.

The data lives in ``data/csv/table5_ablation_source.csv`` as the
single source of truth. This script reads the CSV, copies it to
``data/csv/table5_ablation.csv`` (the committed canonical form)
and emits a Markdown rendering at ``docs/Table5.md``.

Key metric: Debris Suppression Rate on the Wind_Debris_Augmented
sequence, comparing the baseline (MOG2 + CV-KF only) to the
proposed pipeline (MOG2 + CV-KF + Temporal Variance Filter).

Outputs:
    data/csv/table5_ablation.csv
    docs/Table5.md
"""

import csv
import shutil
from pathlib import Path
from typing import Dict


THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent
SOURCE_CSV = REPO / "data" / "csv" / "table5_ablation_source.csv"
TARGET_CSV = REPO / "data" / "csv" / "table5_ablation.csv"
MARKDOWN = REPO / "docs" / "Table5.md"


def load_table(path: Path) -> list:
    with open(path, "r", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(rows: list, out_path: Path) -> None:
    """Re-emit the table in a deterministic column order."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "Configuration",
        "Wind_Debris_Detected_Tracks",
        "Debris_Suppression_Rate_Percent",
        "Culex_LocA_Percent",
    ]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in rows:
            tracks = int(r["Wind_Debris_Detected_Tracks"])
            rate = float(r["Debris_Suppression_Rate_Percent"])
            loca = float(r["Culex_LocA_Percent"])
            w.writerow({
                "Configuration": r["Configuration"],
                "Wind_Debris_Detected_Tracks": tracks,
                "Debris_Suppression_Rate_Percent": f"{rate:.1f}",
                "Culex_LocA_Percent": f"{loca:.2f}",
            })
    print(f"Wrote {out_path}")


def write_markdown(rows: list, out_path: Path) -> None:
    lines = []
    lines.append(
        "# Table 5. Ablation of the Size-Velocity Temporal Variance Filter "
        "on Non-Biological Debris"
    )
    lines.append("")
    lines.append(
        "Evaluation sequence: Wind_Debris_Augmented. The temporal "
        "variance filter is the on-line component that suppresses "
        "wind-blown debris while preserving biological targets "
        "(mosquitoes, fruit flies)."
    )
    lines.append("")
    lines.append(
        "| Configuration | Wind Debris Detected Tracks | "
        "Debris Suppression Rate (%) | Culex LocA (%) |"
    )
    lines.append("|:---|:---:|:---:|:---:|")
    for r in rows:
        tracks = int(r["Wind_Debris_Detected_Tracks"])
        rate = float(r["Debris_Suppression_Rate_Percent"])
        loca = float(r["Culex_LocA_Percent"])
        lines.append(
            f"| {r['Configuration']} | {tracks} | {rate:.1f} | {loca:.2f} |"
        )
    lines.append("")
    lines.append(
        "Adding the temporal variance filter raises the debris "
        "suppression rate from 15.8 % to 89.1 % on the Wind_Debris_Aug "
        "sequence while preserving the Culex_Transit LocA "
        "(67.15 % -> 67.28 %). This isolates the filter as the dominant "
        "contribution to non-biological debris rejection."
    )
    lines.append("")
    lines.append("Script: `evaluation/Table5__VarianceFilterAblation.py`")
    lines.append("Data:   `data/csv/table5_ablation_source.csv`")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> Dict:
    rows = load_table(SOURCE_CSV)
    write_csv(rows, TARGET_CSV)
    write_markdown(rows, MARKDOWN)
    return {"csv": TARGET_CSV, "markdown": MARKDOWN}


if __name__ == "__main__":
    main()
