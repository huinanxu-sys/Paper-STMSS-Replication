"""
Master runner: execute MOG2+LK, MOG2+SORT, MOG2+IMM baselines on the
test sequences and emit per-frame latency CSVs plus an aggregate
Table 1 row CSV that downstream plotting and table-generation scripts
read directly.

Outputs:
    data/csv/baseline_<name>_latency.csv  -- per-frame latency.
    data/csv/table1_baselines.csv         -- aggregated row.
    data/csv/baseline_<name>_debris.csv   -- debris rejection.

Usage:
    python run_baselines_table1.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from baseline_mog2_lk import run_mog2_lk
from baseline_mog2_sort import run_mog2_sort
from baseline_mog2_imm import run_mog2_imm


DEFAULT_VIDEOS = [
    ("Culex_Transit", "Culex_Transit__flying_mosquito.mp4"),
    ("Aedes_Saccade", "Aedes_Saccade__SuppVideo1.mp4"),
    ("Drosophila_Dense", "Drosophila_Dense__drosophila10.avi"),
    ("LevyTest_alpha05", "LevyTest_alpha05__600f.mp4"),
    ("LevyTest_alpha10", "LevyTest_alpha10__600f.mp4"),
    ("LevyTest_alpha15", "LevyTest_alpha15__600f.mp4"),
    ("Synthetic_Swarm", "Synthetic_Swarm__stress_test.mp4"),
    ("Wind_Debris_Aug", "Wind_Debris__augmented.mp4"),
]

VIDEO_DIR = THIS_DIR.parent / "08_Sample_Videos"
DATA_DIR = THIS_DIR.parent / "data" / "csv"

RUNNERS = {
    "MOG2_LK": run_mog2_lk,
    "MOG2_SORT": run_mog2_sort,
    "MOG2_IMM": run_mog2_imm,
}


def run_all(videos=DEFAULT_VIDEOS, max_frames=None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, fname in videos:
        video_path = VIDEO_DIR / fname
        if not video_path.exists():
            print(f"[skip] missing video: {video_path}")
            continue
        for label, runner in RUNNERS.items():
            try:
                t0 = time.perf_counter()
                latencies, _ = runner(str(video_path), max_frames=max_frames)
                dt = time.perf_counter() - t0
            except Exception as exc:
                print(f"[fail] {label} on {fname}: {exc}")
                continue
            if len(latencies) == 0:
                continue
            mean = float(np.mean(latencies))
            p95 = float(np.percentile(latencies, 95))
            std = float(np.std(latencies))
            row_csv = DATA_DIR / f"baseline_{name}_{label}_latency.csv"
            with open(row_csv, 'w', newline='') as f:
                f.write('Frame,Latency_ms\n')
                for i, lat in enumerate(latencies):
                    f.write(f'{i+1},{lat:.4f}\n')
            print(
                f"[ok] {label:<10} {name:<22} mean={mean:6.2f}ms p95={p95:6.2f}ms"
                f" frames={len(latencies)} ({dt:.1f}s total)"
            )
            rows.append({
                'Sequence': name,
                'Pipeline': label,
                'Mean_ms': mean,
                'Std_ms': std,
                'P95_ms': p95,
                'Frames': len(latencies),
            })

    if not rows:
        print("No rows produced. Ensure videos exist in 08_Sample_Videos/.")
        return

    aggregate = {}
    for row in rows:
        aggregate.setdefault(row['Pipeline'], []).append(row)
    out = DATA_DIR / "table1_baselines.csv"
    with open(out, 'w', newline='') as f:
        f.write('Pipeline,Sequence,Mean_ms,Std_ms,P95_ms,Frames\n')
        for row in rows:
            f.write(
                f"{row['Pipeline']},{row['Sequence']},"
                f"{row['Mean_ms']:.3f},{row['Std_ms']:.3f},"
                f"{row['P95_ms']:.3f},{row['Frames']}\n"
            )
    print(f"\nWrote aggregate: {out}")
    print('\nAggregate mean latency across all sequences:')
    for pipeline, items in aggregate.items():
        avg = np.mean([it['Mean_ms'] for it in items])
        p95 = np.mean([it['P95_ms'] for it in items])
        print(f"  {pipeline:<10} avg-mean={avg:6.2f}ms  avg-p95={p95:6.2f}ms")


def main():
    parser = argparse.ArgumentParser(description='Run all MOG2 baselines.')
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()
    run_all(max_frames=args.max_frames)


if __name__ == '__main__':
    main()
