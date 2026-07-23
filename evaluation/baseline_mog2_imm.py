"""
MOG2 + IMM (Interacting Multiple Model) baseline implementation.

Lightweight non-semantic edge baseline: MOG2 foreground extraction, then a
2-model IMM (Constant Velocity + High-Maneuverability) per tracked object.

Reference: Blom & Bar-Shalom, 1988; FilterPy IMM implementation.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.kalman.IMM import IMMEstimator


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


def create_cv_filter(dt=1.0):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0., 0., 0., 0.])
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])
    kf.R *= 2.0
    kf.Q *= 0.1
    kf.P *= 100.0
    return kf


def create_saccade_filter(dt=1.0):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0., 0., 0., 0.])
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])
    kf.R *= 2.0
    kf.Q *= 50.0
    kf.P *= 100.0
    return kf


def spawn_imm(center):
    cv_f = create_cv_filter()
    sac_f = create_saccade_filter()
    cv_f.x[:2] = np.asarray(center, dtype=float)
    sac_f.x[:2] = np.asarray(center, dtype=float)
    trans = np.array([[0.95, 0.05], [0.15, 0.85]])
    mu = np.array([0.8, 0.2])
    return IMMEstimator([cv_f, sac_f], mu, trans)


def run_mog2_imm(video_path, gating_distance=30.0, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    bg = make_bg_subtractor()

    active_imms = []
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
        if len(dets) > 0:
            det_centers = np.array(
                [[d[0] + d[2] / 2, d[1] + d[3] / 2] for d in dets]
            )
        else:
            det_centers = np.empty((0, 2))

        for imm in active_imms:
            imm.predict()

        if len(active_imms) > 0 and len(det_centers) > 0:
            imm_preds = np.array([imm.x[:2].ravel() for imm in active_imms])
            dists = np.linalg.norm(
                det_centers[:, None] - imm_preds[None, :], axis=2
            )
            matched_det, matched_imm = set(), set()
            for _ in range(min(len(det_centers), len(active_imms))):
                idx = np.unravel_index(dists.argmin(), dists.shape)
                if dists[idx] < gating_distance:
                    active_imms[idx[1]].update(det_centers[idx[0]])
                    matched_det.add(idx[0])
                    matched_imm.add(idx[1])
                dists[idx[0], :] = 9999.0
                dists[:, idx[1]] = 9999.0
            for i in range(len(det_centers)):
                if i not in matched_det:
                    active_imms.append(spawn_imm(det_centers[i]))
        elif len(det_centers) > 0:
            for det in det_centers:
                active_imms.append(spawn_imm(det))

        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)
        if len(active_imms) > 0:
            tracked = np.array(
                [imm.x[:2].ravel().tolist() + [15, 15] for imm in active_imms]
            )
            detections_per_frame.append(tracked)
        else:
            detections_per_frame.append(np.empty((0, 4), dtype=np.float32))
        frame_idx += 1

    cap.release()
    return np.array(latencies), detections_per_frame


def main():
    parser = argparse.ArgumentParser(description='MOG2 + IMM baseline.')
    parser.add_argument('--video', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    if not Path(args.video).exists():
        raise FileNotFoundError(args.video)

    latencies, _ = run_mog2_imm(args.video, max_frames=args.max_frames)
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

    print('MOG2+IMM latency: mean={:.3f}ms, p95={:.3f}ms, std={:.3f}ms, frames={}'.format(
        mean, p95, std, len(latencies)))


if __name__ == '__main__':
    main()
