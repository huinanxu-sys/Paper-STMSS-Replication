# -*- coding: utf-8 -*-
"""
ISR (Interception Success Rate) Calculator

Logic:
1. For each frame in Ground Truth, load all GT object centers.
2. For the same frame in Tracker output, load all tracker detection centers.
3. For each GT object, check if ANY tracker detection center falls within
   `capture_radius_px` (proxy for 15cm physical capture radius).
4. A successful interception requires spatial match within the window.
5. ISR = (# successfully intercepted GT objects) / (total GT objects across all frames)

Note: We use pixel distance as proxy. For Culex (30x24px), 50px radius is generous.
For Levy particles (6x6px), we use a smaller radius proportional to object size.
"""

import os
import math
from collections import defaultdict


def load_mot_centroids(filepath):
    """Load MOT file into {frame_id: [(track_id, cx, cy, w, h), ...]}"""
    data = defaultdict(list)
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
                cx = x + w / 2.0
                cy = y + h / 2.0
                data[frame_id].append((track_id, cx, cy, w, h))
    return data


def calculate_isr(gt_path, tracker_path, capture_radius_px=50.0):
    """Calculate ISR for a single sequence."""
    if not os.path.exists(gt_path):
        print(f"  [SKIP] GT not found: {gt_path}")
        return None
    if not os.path.exists(tracker_path):
        print(f"  [SKIP] Tracker not found: {tracker_path}")
        return None

    gt_data = load_mot_centroids(gt_path)
    trk_data = load_mot_centroids(tracker_path)

    total_gt_objects = 0
    intercepted = 0

    for frame_id, gt_objects in gt_data.items():
        trk_objects = trk_data.get(frame_id, [])
        for gt_id, gt_cx, gt_cy, gt_w, gt_h in gt_objects:
            total_gt_objects += 1
            # Check if any tracker detection is within capture radius
            for trk_id, trk_cx, trk_cy, trk_w, trk_h in trk_objects:
                dist = math.hypot(gt_cx - trk_cx, gt_cy - trk_cy)
                if dist <= capture_radius_px:
                    intercepted += 1
                    break  # One interception per GT object is enough

    if total_gt_objects == 0:
        return 0.0, 0, 0

    isr = intercepted / total_gt_objects * 100.0
    return isr, intercepted, total_gt_objects


def main():
    print("=" * 80)
    print("ISR (Interception Success Rate) - Real Data Report")
    print("=" * 80)
    print()

    sequences = [
        ("Culex_Transit", 50.0),           # Real mosquito, ~30x24px
        ("Aedes_Saccade", 50.0),           # Real mosquito, ~35x35px
        ("Drosophila_Dense", 35.0),        # Dense fruit fly, ~15x15px
        ("levy_test_3objects_alpha05_600f", 25.0),  # Synthetic 6x6px
        ("levy_test_3objects_alpha10_600f", 25.0),
        ("levy_test_3objects_alpha15_600f", 25.0),
        ("synthetic_swarm_stress_test", 30.0),
        ("wind_debris_augmented", 40.0),
    ]

    gt_dir = "data/ground_truth"
    trk_dir = "data/table2_stmss_outputs"

    results = []

    for seq_name, radius in sequences:
        gt_path = os.path.join(gt_dir, f"{seq_name}.txt")
        trk_path = os.path.join(trk_dir, f"{seq_name}_STMSS.txt")

        print(f"Processing: {seq_name} (radius={radius}px)")
        result = calculate_isr(gt_path, trk_path, capture_radius_px=radius)
        if result:
            isr, intercepted, total = result
            results.append((seq_name, isr, intercepted, total, radius))
            print(f"  ISR: {isr:.2f}% ({intercepted}/{total})")
        print()

    print("=" * 80)
    print("ISR SUMMARY")
    print("=" * 80)
    print(f"{'Sequence':<40} {'Radius':>8} {'ISR':>10} {'Intercepted':>12} {'Total GT':>10}")
    print("-" * 80)
    for seq_name, isr, intercepted, total, radius in results:
        print(f"{seq_name:<40} {radius:>8.1f} {isr:>9.2f}% {intercepted:>12} {total:>10}")
    print("=" * 80)

    # Save report
    report_path = "ISR_REPORT.txt"
    with open(report_path, 'w') as f:
        f.write("ISR (Interception Success Rate) Report\n")
        f.write("=" * 80 + "\n\n")
        f.write("Method: For each GT object per frame, check if any tracker detection\n")
        f.write("center falls within capture_radius_px of GT center.\n\n")
        f.write(f"{'Sequence':<40} {'Radius':>8} {'ISR':>10} {'Intercepted':>12} {'Total GT':>10}\n")
        f.write("-" * 80 + "\n")
        for seq_name, isr, intercepted, total, radius in results:
            f.write(f"{seq_name:<40} {radius:>8.1f} {isr:>9.2f}% {intercepted:>12} {total:>10}\n")
        f.write("=" * 80 + "\n")

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
