# -*- coding: utf-8 -*-
"""
Adaptive Parameter Tuning System.

Automatically adjusts tracking parameters based on video characteristics
(target type, resolution, frame rate, brightness, motion intensity).
This module is a reference implementation that produces the parameter
templates used as the prior for the offline PG-GA optimisation.

All identifiers, comments, and report strings are in English to match
the EAAI journal submission requirements.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TargetType(Enum):
    """Enumeration of biological target categories."""
    MOSQUITO = "mosquito"       # mosquito: high-speed small target
    ANT = "ant"                 # ant: medium-speed small target
    DROSOPHILA = "drosophila"   # fruit fly: high-speed small target
    MOUSE = "mouse"             # mouse: medium/low-speed large target
    UNKNOWN = "unknown"         # unknown target


@dataclass
class VideoCharacteristics:
    """Container for the measured characteristics of a video stream."""
    resolution: Tuple[int, int]
    fps: float
    brightness: float
    motion_intensity: float
    estimated_target_size: str
    target_type: TargetType


@dataclass
class TrackingParameters:
    """Container for the tuned tracking parameters."""
    # Detection parameters
    min_contour_area: int
    max_contour_area: int

    # Tracking parameters
    max_disappeared: int
    max_distance: int

    # Kalman filter parameters
    kalman_process_noise: float
    kalman_measurement_noise: float

    # Background subtraction parameters
    bg_history: int
    bg_var_threshold: int

    # Levy flight parameters
    levy_alpha: float
    min_trajectory_length: int

    # Confidence threshold
    confidence_threshold: float

    # Low-light enhancement parameters
    low_light_threshold: int
    clahe_clip_limit: float

    def to_dict(self) -> Dict:
        return {
            'min_contour_area': self.min_contour_area,
            'max_contour_area': self.max_contour_area,
            'max_disappeared': self.max_disappeared,
            'max_distance': self.max_distance,
            'kalman_process_noise': self.kalman_process_noise,
            'kalman_measurement_noise': self.kalman_measurement_noise,
            'bg_history': self.bg_history,
            'bg_var_threshold': self.bg_var_threshold,
            'levy_alpha': self.levy_alpha,
            'min_trajectory_length': self.min_trajectory_length,
            'confidence_threshold': self.confidence_threshold,
            'low_light_threshold': self.low_light_threshold,
            'clahe_clip_limit': self.clahe_clip_limit
        }


class AdaptiveParameterTuner:
    """
    Adaptive parameter tuner.

    Analyses a video stream and automatically selects optimal tracking
    parameters based on the inferred target type and observed scene
    statistics (brightness, motion intensity, resolution, fps).
    """

    # Pre-defined parameter templates per target type
    PARAMETER_TEMPLATES = {
        TargetType.MOSQUITO: {
            'min_contour_area': 20,
            'max_contour_area': 200,
            'max_disappeared': 15,
            'max_distance': 100,
            'kalman_process_noise': 0.05,
            'kalman_measurement_noise': 5,
            'bg_history': 300,
            'bg_var_threshold': 12,
            'levy_alpha': 1.3,
            'min_trajectory_length': 10,
            'confidence_threshold': 0.75,
            'low_light_threshold': 60,
            'clahe_clip_limit': 3.0
        },
        TargetType.ANT: {
            'min_contour_area': 30,
            'max_contour_area': 300,
            'max_disappeared': 25,
            'max_distance': 80,
            'kalman_process_noise': 0.03,
            'kalman_measurement_noise': 8,
            'bg_history': 500,
            'bg_var_threshold': 16,
            'levy_alpha': 1.5,
            'min_trajectory_length': 15,
            'confidence_threshold': 0.70,
            'low_light_threshold': 55,
            'clahe_clip_limit': 2.5
        },
        TargetType.DROSOPHILA: {
            'min_contour_area': 25,
            'max_contour_area': 250,
            'max_disappeared': 12,
            'max_distance': 120,
            'kalman_process_noise': 0.06,
            'kalman_measurement_noise': 4,
            'bg_history': 250,
            'bg_var_threshold': 10,
            'levy_alpha': 1.2,
            'min_trajectory_length': 8,
            'confidence_threshold': 0.72,
            'low_light_threshold': 65,
            'clahe_clip_limit': 3.5
        },
        TargetType.MOUSE: {
            'min_contour_area': 200,
            'max_contour_area': 2000,
            'max_disappeared': 40,
            'max_distance': 150,
            'kalman_process_noise': 0.02,
            'kalman_measurement_noise': 15,
            'bg_history': 800,
            'bg_var_threshold': 20,
            'levy_alpha': 1.8,
            'min_trajectory_length': 20,
            'confidence_threshold': 0.65,
            'low_light_threshold': 50,
            'clahe_clip_limit': 2.0
        },
        TargetType.UNKNOWN: {
            'min_contour_area': 50,
            'max_contour_area': 500,
            'max_disappeared': 20,
            'max_distance': 100,
            'kalman_process_noise': 0.03,
            'kalman_measurement_noise': 10,
            'bg_history': 500,
            'bg_var_threshold': 16,
            'levy_alpha': 1.5,
            'min_trajectory_length': 15,
            'confidence_threshold': 0.70,
            'low_light_threshold': 60,
            'clahe_clip_limit': 3.0
        }
    }

    def __init__(self):
        self.video_chars: Optional[VideoCharacteristics] = None
        self.base_params: Optional[TrackingParameters] = None

    def analyze_video(self, video_path: str, sample_frames: int = 30) -> VideoCharacteristics:
        """
        Analyse the visual statistics of an input video.

        Args:
            video_path: path to the input video file.
            sample_frames: number of frames to sample for the analysis.

        Returns:
            VideoCharacteristics describing the observed stream.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Basic video metadata
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Brightness and motion estimation buffers
        brightness_values = []
        motion_scores = []

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError(f"Cannot read video frames: {video_path}")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(prev_gray.mean())

        frame_step = max(1, total_frames // sample_frames)
        frame_count = 1

        while frame_count < sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_step)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append(gray.mean())

            # Dense optical flow for motion intensity
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            motion = cv2.norm(flow, cv2.NORM_L2)
            motion_scores.append(motion)

            prev_gray = gray
            frame_count += 1

        cap.release()

        avg_brightness = np.mean(brightness_values)
        avg_motion = np.mean(motion_scores) if motion_scores else 0

        # Automatic target type inference
        target_type = self._detect_target_type(video_path, avg_motion, avg_brightness)

        # Estimated target size in pixels
        est_size = self._estimate_target_size(target_type, (width, height))

        self.video_chars = VideoCharacteristics(
            resolution=(width, height),
            fps=fps,
            brightness=avg_brightness,
            motion_intensity=avg_motion,
            estimated_target_size=est_size,
            target_type=target_type
        )

        return self.video_chars

    def _detect_target_type(self, video_path: str, motion: float, brightness: float) -> TargetType:
        """Infer the target type from filename cues and motion statistics."""
        path_lower = video_path.lower()

        if 'mosquito' in path_lower or 'fly' in path_lower:
            return TargetType.MOSQUITO
        elif 'ant' in path_lower:
            return TargetType.ANT
        elif 'drosophila' in path_lower:
            return TargetType.DROSOPHILA
        elif 'mice' in path_lower or 'mouse' in path_lower:
            return TargetType.MOUSE
        else:
            # Heuristic fallback based on motion magnitude
            if motion > 400:
                return TargetType.MOUSE
            elif motion > 350:
                return TargetType.DROSOPHILA
            elif motion > 300:
                return TargetType.MOSQUITO
            else:
                return TargetType.ANT

    def _estimate_target_size(self, target_type: TargetType, resolution: Tuple[int, int]) -> str:
        """Estimate the expected target size in pixels for the given type."""
        size_map = {
            TargetType.MOSQUITO: "10-50 px",
            TargetType.ANT: "20-80 px",
            TargetType.DROSOPHILA: "15-60 px",
            TargetType.MOUSE: "100-500 px",
            TargetType.UNKNOWN: "50-200 px"
        }
        return size_map.get(target_type, "unknown")

    def get_optimal_parameters(self, video_path: str = None) -> TrackingParameters:
        """
        Retrieve the tuned tracking parameters.

        Args:
            video_path: optional path used to lazily analyse the video.

        Returns:
            TrackingParameters tuned to the observed stream.
        """
        if self.video_chars is None and video_path is not None:
            self.analyze_video(video_path)

        if self.video_chars is None:
            raise ValueError("Please analyse a video first or provide a video path.")

        # Retrieve the base parameter template
        base_params = self.PARAMETER_TEMPLATES[self.video_chars.target_type].copy()

        # Fine-tune based on the observed characteristics
        adjusted_params = self._fine_tune_parameters(base_params)

        self.base_params = TrackingParameters(**adjusted_params)
        return self.base_params

    def _fine_tune_parameters(self, base_params: Dict) -> Dict:
        """Fine-tune the base parameter set using the observed video statistics."""
        params = base_params.copy()
        chars = self.video_chars

        # 1. Adjust detection area thresholds by resolution
        width, height = chars.resolution
        resolution_scale = np.sqrt(width * height) / np.sqrt(608 * 342)  # relative to reference

        params['min_contour_area'] = int(params['min_contour_area'] * resolution_scale)
        params['max_contour_area'] = int(params['max_contour_area'] * resolution_scale * resolution_scale)

        # 2. Adjust tracking parameters by frame rate
        fps_scale = chars.fps / 24.0  # relative to 24 fps
        params['max_disappeared'] = int(params['max_disappeared'] * fps_scale)
        params['max_distance'] = int(params['max_distance'] * fps_scale)

        # 3. Adjust low-light parameters by brightness
        if chars.brightness < 60:
            # Low-light environment
            params['low_light_threshold'] = int(chars.brightness * 0.8)
            params['clahe_clip_limit'] = min(4.0, params['clahe_clip_limit'] + 0.5)
            params['min_contour_area'] = int(params['min_contour_area'] * 0.7)
        elif chars.brightness > 200:
            # Bright environment
            params['low_light_threshold'] = int(chars.brightness * 0.9)
            params['bg_var_threshold'] = int(params['bg_var_threshold'] * 1.2)

        # 4. Adjust by motion intensity
        if chars.motion_intensity > 500:
            # High-motion scenario
            params['kalman_process_noise'] *= 1.2
            params['max_distance'] = int(params['max_distance'] * 1.3)
        elif chars.motion_intensity < 200:
            # Low-motion scenario
            params['kalman_process_noise'] *= 0.8
            params['max_disappeared'] = int(params['max_disappeared'] * 1.2)

        # 5. Adjust Levy parameters by target type
        if chars.target_type in [TargetType.MOSQUITO, TargetType.DROSOPHILA]:
            # High-speed flying insects: stricter trajectory requirements
            params['min_trajectory_length'] = max(5, int(params['min_trajectory_length'] * 0.8))
            params['levy_alpha'] = max(1.0, params['levy_alpha'] - 0.1)

        return params

    def generate_parameter_report(self) -> str:
        """Generate a human-readable parameter tuning report."""
        if self.video_chars is None or self.base_params is None:
            return "Please analyse a video and obtain parameters first."

        report = []
        report.append("=" * 80)
        report.append("Adaptive Parameter Tuning Report")
        report.append("=" * 80)

        report.append("\n[Video Characteristics Analysis]")
        report.append(f"  Resolution:    {self.video_chars.resolution[0]}x{self.video_chars.resolution[1]}")
        report.append(f"  Frame rate:    {self.video_chars.fps:.2f} FPS")
        report.append(f"  Mean brightness: {self.video_chars.brightness:.1f}")
        report.append(f"  Motion intensity: {self.video_chars.motion_intensity:.2f}")
        report.append(f"  Target type:   {self.video_chars.target_type.value}")
        report.append(f"  Estimated target size: {self.video_chars.estimated_target_size}")

        report.append("\n[Optimised Tracking Parameters]")
        params = self.base_params.to_dict()

        report.append("\n  Detection parameters:")
        report.append(f"    - Min contour area: {params['min_contour_area']}")
        report.append(f"    - Max contour area: {params['max_contour_area']}")

        report.append("\n  Tracking parameters:")
        report.append(f"    - Max disappeared frames: {params['max_disappeared']}")
        report.append(f"    - Max matching distance:  {params['max_distance']}")

        report.append("\n  Kalman filter parameters:")
        report.append(f"    - Process noise: {params['kalman_process_noise']:.3f}")
        report.append(f"    - Measurement noise: {params['kalman_measurement_noise']:.1f}")

        report.append("\n  Background subtraction parameters:")
        report.append(f"    - History frames:  {params['bg_history']}")
        report.append(f"    - Variance threshold: {params['bg_var_threshold']}")

        report.append("\n  Levy flight parameters:")
        report.append(f"    - Levy index alpha: {params['levy_alpha']:.2f}")
        report.append(f"    - Min trajectory length: {params['min_trajectory_length']}")

        report.append("\n  Confidence parameters:")
        report.append(f"    - Confidence threshold: {params['confidence_threshold']:.2f}")

        report.append("\n  Low-light enhancement parameters:")
        report.append(f"    - Low-light threshold: {params['low_light_threshold']}")
        report.append(f"    - CLAHE clip limit: {params['clahe_clip_limit']:.1f}")

        report.append("\n" + "=" * 80)

        return "\n".join(report)


# Convenience helpers
def get_adaptive_params(video_path: str) -> Tuple[TrackingParameters, str]:
    """
    Convenience wrapper that returns the tuned parameters and the report.

    Args:
        video_path: path to the input video file.

    Returns:
        (TrackingParameters, report) tuple.
    """
    tuner = AdaptiveParameterTuner()
    tuner.analyze_video(video_path)
    params = tuner.get_optimal_parameters()
    report = tuner.generate_parameter_report()
    return params, report


if __name__ == "__main__":
    # Test entry point
    import sys

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "samples/Flying_Mosquito.mp4"

    print(f"\nAnalysing video: {video_path}\n")

    try:
        params, report = get_adaptive_params(video_path)
        print(report)
    except Exception as e:
        print(f"Error: {e}")
