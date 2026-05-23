# -*- coding: utf-8 -*-
"""
PAD (Pulse Accumulation & Debounce) Filter

Logic:
1. Read raw tracker output (MOT format).
2. Sort by frame number.
3. Apply temporal debounce: if a detection occurs within `debounce_frames`
   of the previous kept detection, discard it as spurious/high-frequency noise.
4. Count raw triggers vs. filtered triggers.
5. Calculate FPR (False Positive Rate) relative to GT frame count.

This simulates hardware-level pulse debouncing for cyber-physical actuation.
"""

import os
from collections import defaultdict


def load_tracker_raw(filepath):
    """Load tracker output into list of (frame_id, track_id, x, y, w, h)."""
    detections = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                frame_id = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                detections.append((frame_id, track_id, x, y, w, h))
    return detections


def apply_pad_filter(detections, debounce_frames=3):
    """
    Apply PAD temporal debounce.
    Keep a detection only if it's at least `debounce_frames` away
    from the last kept detection (per track ID).
    """
    # Sort by frame
    detections_sorted = sorted(detections, key=lambda d: d[0])

    last_kept_frame = {}  # track_id -> last kept frame
    filtered = []

    for frame_id, track_id, x, y, w, h in detections_sorted:
        if track_id not in last_kept_frame:
            # First time seeing this ID
            last_kept_frame[track_id] = frame_id
            filtered.append((frame_id, track_id, x, y, w, h))
        else:
            gap = frame_id - last_kept_frame[track_id]
            if gap >= debounce_frames:
                last_kept_frame[track_id] = frame_id
                filtered.append((frame_id, track_id, x, y, w, h))
            # else: discard as spurious/high-frequency trigger

    return filtered


def count_unique_triggers(detections):
    """Count total unique trigger events (one per detection line)."""
    return len(detections)


def get_gt_frame_count(gt_path):
    """Get total number of frames from GT file (max frame ID)."""
    if not os.path.exists(gt_path):
        return 0
    max_frame = 0
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                frame_id = int(float(parts[0]))
                max_frame = max(max_frame, frame_id)
    return max_frame


def main():
    print("=" * 80)
    print("PAD (Pulse Accumulation & Debounce) Filter - Real Data Report")
    print("=" * 80)
    print()

    # We focus on Culex_Transit which has the massive 1,993 FP problem
    sequences = [
        "Culex_Transit",
        "Aedes_Saccade",
        "Drosophila_Dense",
        "levy_test_3objects_alpha05_600f",
        "levy_test_3objects_alpha10_600f",
        "levy_test_3objects_alpha15_600f",
        "synthetic_swarm_stress_test",
        "wind_debris_augmented",
    ]

    gt_dir = "data/ground_truth"
    trk_dir = "data/table2_stmss_outputs"

    # Debounce: 100ms at 30 FPS = 3 frames
    debounce_frames = 3

    results = []

    for seq_name in sequences:
        gt_path = os.path.join(gt_dir, f"{seq_name}.txt")
        trk_path = os.path.join(trk_dir, f"{seq_name}_STMSS.txt")

        print(f"Processing: {seq_name}")

        if not os.path.exists(trk_path):
            print(f"  [SKIP] Tracker not found")
            continue

        raw_dets = load_tracker_raw(trk_path)
        filtered_dets = apply_pad_filter(raw_dets, debounce_frames=debounce_frames)

        raw_count = count_unique_triggers(raw_dets)
        filtered_count = count_unique_triggers(filtered_dets)
        reduction = (raw_count - filtered_count) / raw_count * 100.0 if raw_count > 0 else 0.0

        gt_frames = get_gt_frame_count(gt_path)
        fpr_raw = raw_count / gt_frames * 100.0 if gt_frames > 0 else 0.0
        fpr_filtered = filtered_count / gt_frames * 100.0 if gt_frames > 0 else 0.0

        results.append((seq_name, raw_count, filtered_count, reduction, fpr_raw, fpr_filtered, gt_frames))

        print(f"  Raw triggers:      {raw_count}")
        print(f"  Filtered triggers: {filtered_count}")
        print(f"  Reduction:         {reduction:.1f}%")
        print(f"  FPR (raw):         {fpr_raw:.2f}%")
        print(f"  FPR (filtered):    {fpr_filtered:.2f}%")
        print()

    print("=" * 80)
    print("PAD FILTER SUMMARY")
    print("=" * 80)
    print(f"{'Sequence':<40} {'Raw':>8} {'Filtered':>10} {'Reduction':>10} {'FPR Raw':>10} {'FPR Filtered':>14} {'GT Frames':>10}")
    print("-" * 80)
    for seq_name, raw_count, filtered_count, reduction, fpr_raw, fpr_filtered, gt_frames in results:
        print(f"{seq_name:<40} {raw_count:>8} {filtered_count:>10} {reduction:>9.1f}% {fpr_raw:>9.2f}% {fpr_filtered:>13.2f}% {gt_frames:>10}")
    print("=" * 80)

    # Save report
    report_path = "PAD_FILTER_REPORT.txt"
    with open(report_path, 'w') as f:
        f.write("PAD (Pulse Accumulation & Debounce) Filter Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Debounce window: {debounce_frames} frames (~100ms @ 30 FPS)\n\n")
        f.write(f"{'Sequence':<40} {'Raw':>8} {'Filtered':>10} {'Reduction':>10} {'FPR Raw':>10} {'FPR Filtered':>14} {'GT Frames':>10}\n")
        f.write("-" * 80 + "\n")
        for seq_name, raw_count, filtered_count, reduction, fpr_raw, fpr_filtered, gt_frames in results:
            f.write(f"{seq_name:<40} {raw_count:>8} {filtered_count:>10} {reduction:>9.1f}% {fpr_raw:>9.2f}% {fpr_filtered:>13.2f}% {gt_frames:>10}\n")
        f.write("=" * 80 + "\n")

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
