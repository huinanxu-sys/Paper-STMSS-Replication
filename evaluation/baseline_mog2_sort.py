"""
MOG2 + SORT baseline implementation.

Lightweight non-semantic edge baseline: MOG2 foreground extraction, then a
minimal self-contained SORT tracker (Kalman + Hungarian + IoU matching).

Reference: Bewley et al., 2016 (SORT).
"""

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def make_bg_subtractor():
    return cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=False
    )


def extract_micro_flier_blobs(frame_gray, bg_subtractor,
                              min_area=3, max_area=300):
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


class KalmanBoxTracker:
    """Minimal SORT Kalman tracker (x, y, s, r) state."""

    _count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = np.array([
            bbox[0] + bbox[2] / 2,
            bbox[1] + bbox[3] / 2,
            bbox[2] * bbox[3],
            bbox[2] / max(bbox[3], 1e-6),
        ]).reshape(4, 1)

        self.time_since_update = 0
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 1

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(np.array([
            bbox[0] + bbox[2] / 2,
            bbox[1] + bbox[3] / 2,
            bbox[2] * bbox[3],
            bbox[2] / max(bbox[3], 1e-6),
        ]).reshape(4, 1))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def get_state(self):
        s = self.kf.x[2] * self.kf.x[3]
        r = self.kf.x[2] / max(self.kf.x[3], 1e-6)
        return np.array([
            self.kf.x[0] - s / 2,
            self.kf.x[1] - s / 2,
            s,
            r,
        ]).reshape(1, 4)


def run_mog2_sort(video_path, max_age=5, min_hits=3, iou_threshold=0.3,
                  max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    bg = make_bg_subtractor()

    trackers = []
    KalmanBoxTracker._count = 0
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
        dets = extract_micro_flier_blobs(gray, bg)

        for t in reversed(trackers):
            t.predict()

        if len(trackers) > 0 and len(dets) > 0:
            t_boxes = np.array([t.get_state()[0] for t in trackers])
            iou_matrix = iou_batch(dets, t_boxes)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            matched, unmatched_trks, unmatched_dets = (
                [],
                set(range(len(trackers))),
                set(range(len(dets))),
            )
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] < iou_threshold:
                    unmatched_trks.add(c)
                    unmatched_dets.add(r)
                else:
                    trackers[c].update(dets[r])
                    unmatched_trks.discard(c)
                    unmatched_dets.discard(r)
        else:
            unmatched_dets = set(range(len(dets)))
            unmatched_trks = set(range(len(trackers)))

        for idx in unmatched_dets:
            trackers.append(KalmanBoxTracker(dets[idx]))

        ret_tracks = []
        for t in reversed(trackers):
            if t.time_since_update > max_age:
                trackers.remove(t)
            elif t.hit_streak >= min_hits or t.time_since_update == 0:
                ret_tracks.append(t.get_state()[0])

        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)
        detections_per_frame.append(
            np.array(ret_tracks) if ret_tracks else np.empty((0, 4), dtype=np.float32)
        )
        frame_idx += 1

    cap.release()
    return np.array(latencies), detections_per_frame


def main():
    parser = argparse.ArgumentParser(description='MOG2 + SORT baseline.')
    parser.add_argument('--video', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    if not Path(args.video).exists():
        raise FileNotFoundError(args.video)

    latencies, _ = run_mog2_sort(args.video, max_frames=args.max_frames)
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

    print('MOG2+SORT latency: mean={:.3f}ms, p95={:.3f}ms, std={:.3f}ms, frames={}'.format(
        mean, p95, std, len(latencies)))


if __name__ == '__main__':
    main()
