# -*- coding: utf-8 -*-
"""
STMSS Integrated Tracker - Complete Implementation.

Spatio-Temporal Motion Saliency System for micro-vector detection
in photon-starved environments. The on-line pipeline is parameterised
by a target_latency_ms configuration value (see ``STMSSConfig``); the
integrated end-to-end latency on the Culex_Transit reference sequence
is reported in ``data/csv/table1_semantic_baselines.csv`` and is not
hardcoded in this file.
"""

import cv2
import numpy as np
import time
import os
from collections import deque, OrderedDict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Import STMSS core components
# Note: Import from stmss_core__PhotonRestoration_PI-EOKF.py using importlib
import importlib.util
import sys

def load_stmss_core():
    """Load STMSS core module from file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    core_file = os.path.join(current_dir, 'stmss_core__PhotonRestoration_PI-EOKF.py')
    
    spec = importlib.util.spec_from_file_location("stmss_core", core_file)
    stmss_core = importlib.util.module_from_spec(spec)
    sys.modules["stmss_core"] = stmss_core
    spec.loader.exec_module(stmss_core)
    
    return (stmss_core.STMSSConfig, stmss_core.PhotonStarvedRestoration,
            stmss_core.PhysicsInformedEOKF, stmss_core.STMSSMetrics,
            stmss_core.create_stmss_pipeline)

(STMSSConfig, PhotonStarvedRestoration, 
 PhysicsInformedEOKF, STMSSMetrics,
 create_stmss_pipeline) = load_stmss_core()


@dataclass
class STMSSTrackingMetrics:
    """Comprehensive tracking metrics for STMSS"""
    total_frames: int = 0
    total_detections: int = 0
    unique_ids: set = field(default_factory=set)
    active_ids_per_frame: List[int] = field(default_factory=list)
    processing_times: List[float] = field(default_factory=list)
    
    # STMSS-specific
    photon_restoration_stats: List[Dict] = field(default_factory=list)
    pi_eokf_generations: int = 0
    latency_violations: int = 0
    
    # HOTA components
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    id_switches: int = 0


class STMSSKalmanTracker:
    """
    STMSS-optimized Kalman Tracker with PI-EOKF parameters
    """
    
    def __init__(self, initial_position: Tuple[float, float],
                 process_noise: float = 0.03,
                 measurement_noise: float = 10.0):
        
        self.kf = cv2.KalmanFilter(4, 2)
        
        # State transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (observe position only)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # PI-EOKF optimized noise parameters
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * float(process_noise)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(measurement_noise)
        
        # Initial state
        self.kf.statePre = np.array([
            [initial_position[0]],
            [initial_position[1]],
            [0],
            [0]
        ], dtype=np.float32)
        
        self.kf.statePost = self.kf.statePre.copy()
        
        # Tracking metadata
        self.predicted_position = initial_position
        self.last_measurement = initial_position
        self.age = 0
        self.total_visible = 0
        self.consecutive_invisible = 0
        
        # Velocity history for physics constraints
        self.velocity_history = deque(maxlen=5)
        
    def predict(self) -> Tuple[int, int]:
        """Predict next state"""
        predicted = self.kf.predict()
        self.predicted_position = (int(predicted[0]), int(predicted[1]))
        self.age += 1
        return self.predicted_position
    
    def update(self, measurement: Tuple[float, float]) -> Tuple[int, int]:
        """Update with measurement"""
        self.kf.correct(np.array([[measurement[0]], [measurement[1]]], dtype=np.float32))
        self.last_measurement = measurement
        self.total_visible += 1
        self.consecutive_invisible = 0
        
        # Update velocity history
        velocity = (
            self.kf.statePost[2, 0],
            self.kf.statePost[3, 0]
        )
        self.velocity_history.append(velocity)
        
        return (int(self.kf.statePost[0]), int(self.kf.statePost[1]))
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate"""
        return (self.kf.statePost[2, 0], self.kf.statePost[3, 0])
    
    def get_state(self) -> Tuple[int, int]:
        """Get current state estimate"""
        return (int(self.kf.statePost[0]), int(self.kf.statePost[1]))


class STMSSTracker:
    """
    Complete STMSS Tracker
    
    Integrates:
    1. Photon-Starved Signal Restoration
    2. PI-EOKF optimized Kalman filtering
    3. Social Force collision avoidance
    4. Real-time performance monitoring
    """
    
    def __init__(self, config: STMSSConfig = None):
        self.config = config or STMSSConfig()
        
        # STMSS components
        self.stmss = create_stmss_pipeline(self.config)
        self.photon_restoration = self.stmss['photon_restoration']
        self.pi_eokf = self.stmss['pi_eokf']
        self.metrics_monitor = self.stmss['metrics']
        
        # Tracking state
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.kalman_trackers = OrderedDict()
        self.trajectories = OrderedDict()
        
        # Parameters (will be optimized by PI-EOKF)
        self.max_disappeared = 15
        self.max_distance = 100
        self.process_noise = 0.03
        self.measurement_noise = 10.0
        
        # Background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=12, detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Detection parameters
        self.min_contour_area = 20
        self.max_contour_area = 200
        
        # Performance tracking
        self.frame_count = 0
        self.total_latency = 0.0
        
        # Optimization trigger
        self.optimization_interval = 100  # frames
        self.trajectory_buffer = []
        
    def apply_photon_restoration(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply STMSS photon-starved signal restoration"""
        start = time.perf_counter()
        
        # Check if photon-starved (low light)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 60:  # Photon-starved condition
            # Use full entropy maximization
            restored, stats = self.photon_restoration.maximize_entropy(frame)
        else:
            # Use fast version
            restored = self.photon_restoration.fast_photon_recovery(frame)
            stats = {'fast_mode': True, 'brightness': mean_brightness}
        
        elapsed = (time.perf_counter() - start) * 1000
        stats['restoration_time_ms'] = elapsed
        
        return restored, stats
    
    def detect_motion(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """STMSS-optimized motion detection"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0.005)
        
        # Threshold and morphological operations
        _, fg_mask = cv2.threshold(fg_mask, 150, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area <= area <= self.max_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, x + w, y + h))
        
        return bboxes
    
    def calculate_social_forces(self, positions: List[Tuple[int, int]]) -> np.ndarray:
        """Calculate social force repulsion between objects"""
        n = len(positions)
        if n < 2:
            return np.zeros((n, 2))
        
        forces = np.zeros((n, 2))
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = np.array(positions[i]) - np.array(positions[j])
                distance = np.linalg.norm(diff)
                
                if distance < self.config.social_force_radius and distance > 0:
                    # Repulsion force
                    force_magnitude = self.config.repulsion_strength * np.exp(
                        -distance / (self.config.social_force_radius * 0.3)
                    )
                    force_direction = diff / distance
                    
                    forces[i] += force_direction * force_magnitude
                    forces[j] -= force_direction * force_magnitude
        
        return forces
    
    def update(self, bboxes: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int]]:
        """
        Update tracker with new detections
        
        Implements STMSS tracking with PI-EOKF and social forces
        """
        # Calculate centroids
        input_centroids = []
        for (startX, startY, endX, endY) in bboxes:
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids.append((cX, cY))
        
        # If no existing objects, register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self._register(input_centroids[i], bboxes[i] if i < len(bboxes) else None)
            return self.objects
        
        # Calculate social forces for existing objects
        existing_positions = list(self.objects.values())
        social_forces = self.calculate_social_forces(existing_positions)
        
        # Predict with Kalman and apply social forces
        predicted_positions = []
        for i, (object_id, _) in enumerate(self.objects.items()):
            if object_id in self.kalman_trackers:
                predicted = self.kalman_trackers[object_id].predict()
                
                # Apply social force correction
                if i < len(social_forces):
                    corrected = (
                        int(predicted[0] + social_forces[i][0]),
                        int(predicted[1] + social_forces[i][1])
                    )
                else:
                    corrected = predicted
                
                predicted_positions.append(corrected)
            else:
                predicted_positions.append(self.objects[object_id])
        
        # Calculate distance matrix
        D = self._distance_matrix(predicted_positions, input_centroids)
        
        # Get object IDs list for indexing
        object_ids = list(self.objects.keys())
        
        # Hungarian algorithm for optimal assignment
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(D)
        
        used_rows = set()
        used_cols = set()
        
        # Update matched objects
        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            
            if row >= len(object_ids) or col >= len(input_centroids):
                continue
            
            if D[row, col] > self.max_distance:
                continue
            
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            
            # Update Kalman filter
            if object_id in self.kalman_trackers:
                self.kalman_trackers[object_id].update(input_centroids[col])
            
            # Update trajectory
            self.trajectories[object_id].append(input_centroids[col])
            
            used_rows.add(row)
            used_cols.add(col)
        
        # Handle disappeared objects
        unused_rows = set(range(D.shape[0])) - used_rows
        unused_cols = set(range(D.shape[1])) - used_cols
        
        for row in unused_rows:
            if row >= len(object_ids):
                continue
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            
            # Use Kalman prediction during occlusion
            if object_id in self.kalman_trackers:
                predicted = self.kalman_trackers[object_id].predict()
                self.objects[object_id] = predicted
                self.trajectories[object_id].append(predicted)
            
            if self.disappeared[object_id] > self.max_disappeared:
                self._deregister(object_id)
        
        # Register new objects
        for col in unused_cols:
            self._register(input_centroids[col], bboxes[col] if col < len(bboxes) else None)
        
        # Collect trajectory data for PI-EOKF optimization
        self._collect_trajectory_data()
        
        # Periodic PI-EOKF optimization
        if self.frame_count % self.optimization_interval == 0 and len(self.trajectory_buffer) > 10:
            self._optimize_parameters()
        
        return self.objects
    
    def _distance_matrix(self, predicted: List[Tuple[int, int]], 
                        detected: List[Tuple[int, int]]) -> np.ndarray:
        """Calculate distance matrix between predicted and detected positions"""
        D = np.zeros((len(predicted), len(detected)))
        
        for i, pred in enumerate(predicted):
            for j, det in enumerate(detected):
                D[i, j] = np.sqrt((pred[0] - det[0])**2 + (pred[1] - det[1])**2)
        
        return D
    
    def _register(self, centroid: Tuple[int, int], bbox: Optional[Tuple[int, int, int, int]]):
        """Register new object"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.trajectories[self.next_object_id] = deque(maxlen=50)
        self.trajectories[self.next_object_id].append(centroid)
        
        # Create Kalman tracker with PI-EOKF parameters
        self.kalman_trackers[self.next_object_id] = STMSSKalmanTracker(
            centroid,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise
        )
        
        self.next_object_id += 1
    
    def _deregister(self, object_id: int):
        """Deregister object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.kalman_trackers:
            del self.kalman_trackers[object_id]
        if object_id in self.trajectories:
            del self.trajectories[object_id]
    
    def _collect_trajectory_data(self):
        """Collect trajectory data for PI-EOKF optimization"""
        for obj_id, trajectory in self.trajectories.items():
            if len(trajectory) >= 3:
                positions = list(trajectory)[-3:]
                velocities = []
                accelerations = []
                
                for i in range(1, len(positions)):
                    v = (positions[i][0] - positions[i-1][0],
                         positions[i][1] - positions[i-1][1])
                    velocities.append(v)
                
                for i in range(1, len(velocities)):
                    a = (velocities[i][0] - velocities[i-1][0],
                         velocities[i][1] - velocities[i-1][1])
                    accelerations.append(a)
                
                self.trajectory_buffer.append({
                    'positions': positions,
                    'velocities': velocities,
                    'accelerations': accelerations
                })
    
    def _optimize_parameters(self):
        """Run PI-EOKF optimization"""
        print(f"[STMSS] Running PI-EOKF optimization at frame {self.frame_count}...")
        
        optimized = self.pi_eokf.optimize(self.trajectory_buffer)
        
        # Update tracker parameters
        self.process_noise = optimized['process_noise']
        self.measurement_noise = optimized['measurement_noise']
        
        # Update existing Kalman filters
        for tracker in self.kalman_trackers.values():
            tracker.kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.process_noise
            tracker.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.measurement_noise
        
        # Clear buffer
        self.trajectory_buffer = []
        
        print(f"[STMSS] Parameters updated: Q={self.process_noise:.4f}, R={self.measurement_noise:.2f}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Dict, np.ndarray, Dict]:
        """
        Process single frame with complete STMSS pipeline
        
        Returns:
            objects: Tracked objects
            visualization: Annotated frame
            stats: Performance statistics
        """
        self.metrics_monitor.start_frame()
        overall_start = time.perf_counter()
        
        # Stage 1: Photon-Starved Signal Restoration
        restored, restoration_stats = self.apply_photon_restoration(frame)
        
        # Stage 2: Motion Detection
        bboxes = self.detect_motion(restored)
        
        # Stage 3: STMSS Tracking with PI-EOKF and Social Forces
        objects = self.update(bboxes)
        
        # Stage 4: Visualization
        vis_frame = self._visualize(frame, objects, bboxes)
        
        # Performance metrics
        self.metrics_monitor.end_frame()
        overall_time = (time.perf_counter() - overall_start) * 1000
        
        self.frame_count += 1
        self.total_latency += overall_time
        
        # Check latency constraint
        latency_violation = overall_time > self.config.target_latency_ms
        
        stats = {
            'latency_ms': overall_time,
            'restoration_stats': restoration_stats,
            'num_detections': len(bboxes),
            'num_tracks': len(objects),
            'latency_violation': latency_violation,
            'performance': self.metrics_monitor.get_current_stats()
        }
        
        return objects, vis_frame, stats
    
    def _visualize(self, frame: np.ndarray, objects: Dict, bboxes: List) -> np.ndarray:
        """Create visualization overlay"""
        vis = frame.copy()
        
        # Draw bounding boxes
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw tracked objects
        for obj_id, (cx, cy) in objects.items():
            color = (0, 0, 255) if obj_id % 2 == 0 else (255, 0, 0)
            cv2.circle(vis, (cx, cy), 5, color, -1)
            cv2.putText(vis, f"ID:{obj_id}", (cx + 10, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw trajectory
            if obj_id in self.trajectories:
                traj = list(self.trajectories[obj_id])
                for i in range(1, len(traj)):
                    cv2.line(vis, traj[i-1], traj[i], color, 2)
        
        # Draw performance info
        stats = self.metrics_monitor.get_current_stats()
        info_text = f"STMSS | Latency: {stats['current_latency_ms']:.1f}ms | FPS: {stats['current_fps']:.1f}"
        cv2.putText(vis, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        avg_latency = self.total_latency / self.frame_count if self.frame_count > 0 else 0
        
        return {
            'total_frames': self.frame_count,
            'total_tracks': self.next_object_id,
            'avg_latency_ms': avg_latency,
            'target_latency_ms': self.config.target_latency_ms,
            'target_met': avg_latency < self.config.target_latency_ms,
            'performance': self.metrics_monitor.get_current_stats()
        }


def test_stmss_tracker():
    """Test STMSS tracker with synthetic data"""
    print("=" * 80)
    print("STMSS Tracker Test")
    print("=" * 80)
    print("\nNOTE: This test uses purely synthetic circle images and is intended")
    print("      to validate the Pipeline API connectivity and latency logic only.")
    print("      Production-grade results are produced by the high-density")
    print("      agricultural video benchmark, not by this test.")

    # Use the configuration's default target_latency_ms; do not hardcode
    # any value here. The test prints the threshold from the config so
    # the synthetic-frame result is interpretable without coupling the
    # source code to any specific quantitative target in the paper.
    config = STMSSConfig()
    tracker = STMSSTracker(config)
    target_ms = config.target_latency_ms

    # Test with synthetic video
    print("\n[1] Testing with synthetic frames...")

    for i in range(50):
        # Create synthetic frame with moving object
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add moving circle (simulating micro-vector)
        cx = 100 + i * 5
        cy = 200 + int(50 * np.sin(i * 0.2))
        cv2.circle(frame, (cx, cy), 10, (200, 200, 200), -1)

        # Add noise (simulating photon-starved condition)
        noise = np.random.normal(0, 10, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)

        # Process frame
        objects, vis, stats = tracker.process_frame(frame)

        if i % 10 == 0:
            print(f"  Frame {i}: {stats['num_tracks']} tracks, "
                  f"{stats['latency_ms']:.2f}ms latency")

    # Print summary
    summary = tracker.get_summary_stats()
    print("\n[2] Summary Statistics")
    print(f"  Total frames: {summary['total_frames']}")
    print(f"  Total tracks: {summary['total_tracks']}")
    print(f"  Avg latency: {summary['avg_latency_ms']:.2f}ms")
    print(f"  Target T_comp = {target_ms:.2f} ms (config): "
          f"{'MET' if summary['target_met'] else 'NOT MET'}")

    print("\n" + "=" * 80)
    print("STMSS Tracker Test Complete")
    print("=" * 80)


if __name__ == "__main__":
    test_stmss_tracker()
