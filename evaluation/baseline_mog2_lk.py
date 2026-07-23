"""
MOG2 + Lucas-Kanade (LK) baseline implementation.

Lightweight non-semantic edge baseline that uses MOG2 background subtraction
to obtain foreground blobs and Lucas-Kanade sparse optical flow to track the
centroids of those blobs frame-to-frame.

Outputs:
    * Mean / P95 inference latency (excluding video I/O).
    * Per-frame detection bounding boxes in (x, y, w, h) MOT format.
    * Per-frame debris rejection rate (matched against ground-truth class labels).

Reference: Lucas & Kanade, 1981; Zivkovic, 2004.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np


def make_bg_subtractor():
    """MOG2 background subtractor tuned for micro-fliers."""
    return cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=False
    )


def extract_micro_flier_blobs(frame_gray, bg_subtractor,
                              min_area=3, max_area=300):
    """MOG2 -> morphological opening -> contour -> (x, y, w, h)."""
    mask = bg_subtractor.apply(frame_gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])
    return np.array(detections) if detections else np.empty((0, 4), dtype=np.float32)


def iou_batch(bb_test, bb_gt):
    """Vectorised IoU between two bounding-box arrays."""
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.zeros((len(bb_test), len(bb_gt)), dtype=np.float32)
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(
        bb_test[..., 0] + bb_test[..., 2],
        bb_gt[..., 0] + bb_gt[..., 2],
    )
    yy2 = np.minimum(
        bb_test[..., 1] + bb_test[..., 3],
        bb_gt[..., 1] + bb_gt[..., 3],
    )
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    return wh / (
        bb_test[..., 2] * bb_test[..., 3]
        + bb_gt[..., 2] * bb_gt[..., 3]
        - wh
    )


def run_mog2_lk(video_path, max_frames=None):
    """Run MOG2 + Lucas-Kanade and return latencies and detections."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    bg = make_bg_subtractor()
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03
        ),
    )

    prev_gray = None
    p0 = None
    prev_boxes = np.empty((0, 4), dtype=np.float32)
    latencies = []
    detections_per_frame = []

    frame_idx = 0
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        start = time.perf_counter()

        current_boxes = extract_micro_flier_blobs(gray, bg)

        if (
            prev_gray is not None
            and p0 is not None
            and len(p0) > 0
            and len(current_boxes) > 0
        ):
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, p0, None, **lk_params
            )
            if p1 is not None and st is not None:
                good_new = p1[st == 1].reshape(-1, 2)
                good_old = p0[st == 1].reshape(-1, 2)
                if len(good_old) > 0 and len(prev_boxes) > 0:
                    centers = np.column_stack(
                        [
                            prev_boxes[:, 0] + prev_boxes[:, 2] / 2.0,
                            prev_boxes[:, 1] + prev_boxes[:, 3] / 2.0,
                        ]
                    )
                    tracked = []
                    for new_pt, old_pt in zip(good_new, good_old):
                        dists = np.linalg.norm(centers - old_pt, axis=1)
                        if len(dists) > 0 and np.min(dists) < 15.0:
                            idx = int(np.argmin(dists))
                            w, h = prev_boxes[idx, 2], prev_boxes[idx, 3]
                            tracked.append(
                                [float(new_pt[0] - w / 2),
                                 float(new_pt[1] - h / 2),
                                 float(w), float(h)]
                            )
                    if tracked:
                        current_boxes = np.vstack(
                            [current_boxes, np.array(tracked)]
                        )

        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)
        detections_per_frame.append(current_boxes.copy())

        prev_gray = gray
        if len(current_boxes) > 0:
            p0 = np.array(
                [
                    [b[0] + b[2] / 2, b[1] + b[3] / 2]
                    for b in current_boxes
                ],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            prev_boxes = current_boxes
        else:
            p0 = None
            prev_boxes = np.empty((0, 4), dtype=np.float32)
        frame_idx += 1

    cap.release()
    return np.array(latencies), detections_per_frame


def evaluate_debris_rejection(detections, gt_labels_per_frame, iou_threshold=0.3):
    """Fraction of detected tracks that overlap a 'debris' ground truth.

    gt_labels_per_frame: list per frame, each is a list of dicts with
        keys 'box' (x, y, w, h) and 'class' (str).
    Returns a float in [0, 1].
    """
    debris_hits = 0
    total_dets = 0
    for dets, gts in zip(detections, gt_labels_per_frame):
        if len(dets) == 0:
            continue
        gt_boxes = np.array([g['box'] for g in gts]) if gts else np.empty((0, 4))
        gt_classes = [g['class'] for g in gts]
        if len(gt_boxes) == 0:
            total_dets += len(dets)
            continue
        iou = iou_batch(dets, gt_boxes)
        total_dets += len(dets)
        for i in range(len(dets)):
            best = iou[i].max() if iou.size else 0
            if best >= iou_threshold:
                j = int(iou[i].argmax())
                if gt_classes[j] == 'debris':
                    debris_hits += 1
    return (debris_hits / total_dets) if total_dets else 0.0


def main():
    parser = argparse.ArgumentParser(description='MOG2 + Lucas-Kanade baseline.')
    parser.add_argument('--video', required=True, help='Path to input video.')
    parser.add_argument(
        '--csv', required=True, help='Path to write per-frame latency CSV.'
    )
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    if not Path(args.video).exists():
        raise FileNotFoundError(args.video)

    latencies, dets = run_mog2_lk(args.video, max_frames=args.max_frames)
    if len(latencies) == 0:
        raise RuntimeError('No frames processed.')

    mean = float(np.mean(latencies))
    p95 = float(np.percentile(latencies, 95))
    std = float(np.std(latencies))

    os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
    with open(args.csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Latency_ms'])
        for i, lat in enumerate(latencies):
            writer.writerow([i + 1, f'{lat:.4f}'])

    print('MOG2+LK latency: mean={:.3f}ms, p95={:.3f}ms, std={:.3f}ms, frames={}'.format(
        mean, p95, std, len(latencies)))


if __name__ == '__main__':
    main()
