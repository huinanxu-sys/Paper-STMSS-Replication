# -*- coding: utf-8 -*-
"""
自适应参数调优系统
根据视频特征自动调整追踪参数
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TargetType(Enum):
    """目标类型枚举"""
    MOSQUITO = "mosquito"      # 蚊虫：高速小目标
    ANT = "ant"                # 蚂蚁：中速小目标
    DROSOPHILA = "drosophila"  # 果蝇：高速小目标
    MOUSE = "mouse"            # 小鼠：中低速大目标
    UNKNOWN = "unknown"        # 未知类型


@dataclass
class VideoCharacteristics:
    """视频特征数据类"""
    resolution: Tuple[int, int]
    fps: float
    brightness: float
    motion_intensity: float
    estimated_target_size: str
    target_type: TargetType


@dataclass
class TrackingParameters:
    """追踪参数数据类"""
    # 检测参数
    min_contour_area: int
    max_contour_area: int
    
    # 追踪参数
    max_disappeared: int
    max_distance: int
    
    # Kalman滤波参数
    kalman_process_noise: float
    kalman_measurement_noise: float
    
    # 背景减除参数
    bg_history: int
    bg_var_threshold: int
    
    # Lévy飞行参数
    levy_alpha: float
    min_trajectory_length: int
    
    # 置信度阈值
    confidence_threshold: float
    
    # 低光增强参数
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
    自适应参数调优器
    根据视频特征和目标类型自动选择最优参数
    """
    
    # 预设参数模板
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
        self.video_chars = None
        self.base_params = None
        
    def analyze_video(self, video_path: str, sample_frames: int = 30) -> VideoCharacteristics:
        """
        分析视频特征
        
        Args:
            video_path: 视频文件路径
            sample_frames: 采样帧数
            
        Returns:
            VideoCharacteristics: 视频特征
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        # 基本视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 分析亮度和运动
        brightness_values = []
        motion_scores = []
        
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError(f"无法读取视频帧: {video_path}")
        
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
            
            # 计算光流
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
        
        # 自动识别目标类型
        target_type = self._detect_target_type(video_path, avg_motion, avg_brightness)
        
        # 估计目标大小
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
        """根据视频路径和特征检测目标类型"""
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
            # 基于运动特征推断
            if motion > 400:
                return TargetType.MOUSE
            elif motion > 350:
                return TargetType.DROSOPHILA
            elif motion > 300:
                return TargetType.MOSQUITO
            else:
                return TargetType.ANT
    
    def _estimate_target_size(self, target_type: TargetType, resolution: Tuple[int, int]) -> str:
        """估计目标大小"""
        size_map = {
            TargetType.MOSQUITO: "10-50像素",
            TargetType.ANT: "20-80像素",
            TargetType.DROSOPHILA: "15-60像素",
            TargetType.MOUSE: "100-500像素",
            TargetType.UNKNOWN: "50-200像素"
        }
        return size_map.get(target_type, "未知")
    
    def get_optimal_parameters(self, video_path: str = None) -> TrackingParameters:
        """
        获取最优追踪参数
        
        Args:
            video_path: 视频路径（如果之前未分析）
            
        Returns:
            TrackingParameters: 最优参数
        """
        if self.video_chars is None and video_path is not None:
            self.analyze_video(video_path)
        
        if self.video_chars is None:
            raise ValueError("请先分析视频或提供视频路径")
        
        # 获取基础参数模板
        base_params = self.PARAMETER_TEMPLATES[self.video_chars.target_type].copy()
        
        # 根据视频特征微调
        adjusted_params = self._fine_tune_parameters(base_params)
        
        self.base_params = TrackingParameters(**adjusted_params)
        return self.base_params
    
    def _fine_tune_parameters(self, base_params: Dict) -> Dict:
        """根据视频特征微调参数"""
        params = base_params.copy()
        chars = self.video_chars
        
        # 1. 根据分辨率调整检测区域
        width, height = chars.resolution
        resolution_scale = np.sqrt(width * height) / np.sqrt(608 * 342)  # 相对于参考分辨率
        
        # 调整轮廓面积阈值
        params['min_contour_area'] = int(params['min_contour_area'] * resolution_scale)
        params['max_contour_area'] = int(params['max_contour_area'] * resolution_scale * resolution_scale)
        
        # 2. 根据帧率调整追踪参数
        fps_scale = chars.fps / 24.0  # 相对于24fps
        params['max_disappeared'] = int(params['max_disappeared'] * fps_scale)
        params['max_distance'] = int(params['max_distance'] * fps_scale)
        
        # 3. 根据亮度调整低光参数
        if chars.brightness < 60:
            # 低光环境
            params['low_light_threshold'] = int(chars.brightness * 0.8)
            params['clahe_clip_limit'] = min(4.0, params['clahe_clip_limit'] + 0.5)
            params['min_contour_area'] = int(params['min_contour_area'] * 0.7)
        elif chars.brightness > 200:
            # 强光环境
            params['low_light_threshold'] = int(chars.brightness * 0.9)
            params['bg_var_threshold'] = int(params['bg_var_threshold'] * 1.2)
        
        # 4. 根据运动强度调整
        if chars.motion_intensity > 500:
            # 高强度运动
            params['kalman_process_noise'] *= 1.2
            params['max_distance'] = int(params['max_distance'] * 1.3)
        elif chars.motion_intensity < 200:
            # 低强度运动
            params['kalman_process_noise'] *= 0.8
            params['max_disappeared'] = int(params['max_disappeared'] * 1.2)
        
        # 5. 根据目标类型调整Lévy参数
        if chars.target_type in [TargetType.MOSQUITO, TargetType.DROSOPHILA]:
            # 高速飞行昆虫：更严格的轨迹要求
            params['min_trajectory_length'] = max(5, int(params['min_trajectory_length'] * 0.8))
            params['levy_alpha'] = max(1.0, params['levy_alpha'] - 0.1)
        
        return params
    
    def generate_parameter_report(self) -> str:
        """生成参数调优报告"""
        if self.video_chars is None or self.base_params is None:
            return "请先分析视频并获取参数"
        
        report = []
        report.append("=" * 80)
        report.append("自适应参数调优报告")
        report.append("=" * 80)
        
        report.append("\n【视频特征分析】")
        report.append(f"  分辨率: {self.video_chars.resolution[0]}x{self.video_chars.resolution[1]}")
        report.append(f"  帧率: {self.video_chars.fps:.2f} FPS")
        report.append(f"  平均亮度: {self.video_chars.brightness:.1f}")
        report.append(f"  运动强度: {self.video_chars.motion_intensity:.2f}")
        report.append(f"  目标类型: {self.video_chars.target_type.value}")
        report.append(f"  估计目标大小: {self.video_chars.estimated_target_size}")
        
        report.append("\n【优化后的追踪参数】")
        params = self.base_params.to_dict()
        
        report.append("\n  检测参数:")
        report.append(f"    - 最小轮廓面积: {params['min_contour_area']}")
        report.append(f"    - 最大轮廓面积: {params['max_contour_area']}")
        
        report.append("\n  追踪参数:")
        report.append(f"    - 最大消失帧数: {params['max_disappeared']}")
        report.append(f"    - 最大匹配距离: {params['max_distance']}")
        
        report.append("\n  Kalman滤波参数:")
        report.append(f"    - 过程噪声: {params['kalman_process_noise']:.3f}")
        report.append(f"    - 测量噪声: {params['kalman_measurement_noise']:.1f}")
        
        report.append("\n  背景减除参数:")
        report.append(f"    - 历史帧数: {params['bg_history']}")
        report.append(f"    - 方差阈值: {params['bg_var_threshold']}")
        
        report.append("\n  Lévy飞行参数:")
        report.append(f"    - Lévy指数: {params['levy_alpha']:.2f}")
        report.append(f"    - 最小轨迹长度: {params['min_trajectory_length']}")
        
        report.append("\n  置信度参数:")
        report.append(f"    - 置信度阈值: {params['confidence_threshold']:.2f}")
        
        report.append("\n  低光增强参数:")
        report.append(f"    - 低光阈值: {params['low_light_threshold']}")
        report.append(f"    - CLAHE限制: {params['clahe_clip_limit']:.1f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


# 便捷函数
def get_adaptive_params(video_path: str) -> Tuple[TrackingParameters, str]:
    """
    获取视频的自适应参数
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        (TrackingParameters, report): 参数和报告
    """
    tuner = AdaptiveParameterTuner()
    tuner.analyze_video(video_path)
    params = tuner.get_optimal_parameters()
    report = tuner.generate_parameter_report()
    return params, report


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "samples/Flying_Mosquito.mp4"
    
    print(f"\n分析视频: {video_path}\n")
    
    try:
        params, report = get_adaptive_params(video_path)
        print(report)
    except Exception as e:
        print(f"错误: {e}")
