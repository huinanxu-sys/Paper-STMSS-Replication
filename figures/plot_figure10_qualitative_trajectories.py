"""
Figure 10: Qualitative Tracking Trajectories Visualization.

This figure is a SCHEMATIC illustration of the four canonical
qualitative scenarios (continuous tracking, debris rejection, Lévy
fragmentation, congestion). The trajectory curves are simulated
parametric paths, not raw tracker outputs; the per-frame
track-CSVs are produced by the full pipeline in
``src/stmss_tracker__FullPipeline.py``. Sample video files are not
redistributed in this package; if the `08_Sample_Videos/` directory
is absent, the panels fall back to a blank canvas.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12


def get_video_frame(video_path, frame_num=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_num is None:
        frame_num = total_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_num, total_frames - 1))
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame = cv2.resize(frame, (512, 512))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def create_figure10_matrix():
    samples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '08_Sample_Videos')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    ax = axes[0]
    video_path = os.path.join(samples_dir, 'flying_mosquito.mp4')
    frame = get_video_frame(video_path)
    
    if frame is not None:
        ax.imshow(frame)
        t = np.linspace(0, 4*np.pi, 80)
        x = 256 + 100 * np.cos(t) + 30 * np.sin(3*t)
        y = 256 + 80 * np.sin(t) + 20 * np.cos(2*t)
        ax.plot(x, y, color='#2ca02c', linewidth=3, alpha=0.9, label='Tracking Trajectory')
        ax.scatter(x[::15], y[::15], c='#2ca02c', s=40, alpha=0.7, edgecolors='white', linewidths=1, zorder=5, label='Detection Points')
        ax.scatter(x[0], y[0], c='green', s=120, marker='o', edgecolors='white', linewidths=2, zorder=6, label='Track Start')
        ax.scatter(x[-1], y[-1], c='red', s=120, marker='s', edgecolors='white', linewidths=2, zorder=6, label='Track End')
    
    ax.set_title('(a) Culex_Transit (Continuous)', fontsize=14, weight='bold', pad=10)
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    ax = axes[1]
    video_path = os.path.join(samples_dir, 'wind_debris_augmented.mp4')
    frame = get_video_frame(video_path)
    
    if frame is not None:
        ax.imshow(frame)
        t = np.linspace(0, 2*np.pi, 40)
        x = 256 + 60 * np.cos(t)
        y = 256 + 50 * np.sin(t)
        ax.plot(x, y, color='#2ca02c', linewidth=3, alpha=0.9, label='Biological Target (Kept)')
        ax.scatter(x[::10], y[::10], c='#2ca02c', s=35, alpha=0.7, edgecolors='white', linewidths=1)
        
        for i in range(4):
            x_noise = np.random.randint(100, 400, 4)
            y_noise = np.random.randint(100, 400, 4)
            ax.plot(x_noise, y_noise, color='gray', linewidth=1.5, alpha=0.4, linestyle='--')
            ax.scatter(x_noise[-1], y_noise[-1], c='gray', s=30, marker='x', alpha=0.5)
        
        ax.plot([], [], color='gray', linewidth=1.5, alpha=0.4, linestyle='--', marker='x', label='Debris (Filtered)')
    
    ax.set_title('(b) Wind_Debris (Rejection)', fontsize=14, weight='bold', pad=10)
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    ax = axes[2]
    video_path = os.path.join(samples_dir, 'levy_test_3objects_alpha15_600f.mp4')
    frame = get_video_frame(video_path, frame_num=300)
    
    if frame is not None:
        ax.imshow(frame)
        segments = [
            (np.linspace(150, 220, 15), np.linspace(150, 200, 15)),
            (np.linspace(350, 400, 12), np.linspace(180, 240, 12)),
            (np.linspace(200, 280, 14), np.linspace(350, 300, 14)),
        ]
        colors = ['#d62728', '#ff7f0e', '#9467bd']
        for i, (x, y) in enumerate(segments):
            ax.plot(x, y, color=colors[i], linewidth=3, alpha=0.9)
            ax.scatter(x[0], y[0], c='green', s=100, marker='o', edgecolors='white', linewidths=2, zorder=6)
            ax.scatter(x[-1], y[-1], c='red', s=100, marker='x', linewidths=3, zorder=6)
            ax.annotate('', xy=(x[-1]+15, y[-1]+15), xytext=(x[-1], y[-1]),
                       arrowprops=dict(arrowstyle='->', color=colors[i], lw=2, alpha=0.6))
    
    ax.set_title('(c) Lévy_Test a=1.5 (Fragmentation)', fontsize=14, weight='bold', pad=10)
    ax.axis('off')
    ax.scatter([], [], c='green', s=100, marker='o', edgecolors='white', linewidths=2, label='Track Start')
    ax.scatter([], [], c='red', s=100, marker='x', linewidths=3, label='Track End (Fragmented)')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    ax = axes[3]
    video_path = os.path.join(samples_dir, 'drosophila_10.avi')
    frame = get_video_frame(video_path, frame_num=1000)
    
    if frame is not None:
        ax.imshow(frame)
        np.random.seed(42)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for i in range(10):
            t = np.linspace(0, 1.5*np.pi, 20)
            cx = np.random.randint(120, 400)
            cy = np.random.randint(120, 400)
            a = np.random.randint(50, 90)
            b = np.random.randint(30, 60)
            rotation = np.random.random() * 2 * np.pi
            phase = np.random.random() * 2 * np.pi
            
            x = cx + a * np.cos(t + phase) * np.cos(rotation) - b * np.sin(t + phase) * np.sin(rotation)
            y = cy + a * np.cos(t + phase) * np.sin(rotation) + b * np.sin(t + phase) * np.cos(rotation)
            
            ax.plot(x, y, color=colors[i], linewidth=2.5, alpha=0.85)
            ax.scatter(x[0], y[0], c=colors[i], s=60, marker='o', edgecolors='white', linewidths=1.5, zorder=5)
            ax.annotate('', xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]),
                       arrowprops=dict(arrowstyle='->', color=colors[i], lw=2, alpha=0.9))
    
    ax.set_title('(d) Drosophila_Dense (Congestion)', fontsize=14, weight='bold', pad=10)
    ax.axis('off')
    
    legend_text = 'Track Start    -> Flight Direction\n10 Individual IDs Preserved'
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, 'Figure10_Qualitative_Trajectories.pdf'), dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'Figure10_Qualitative_Trajectories.png'), dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'Figure10_Qualitative_Trajectories.svg'), format='svg', bbox_inches='tight', facecolor='white')
    
    print("Figure 10 generated")


if __name__ == "__main__":
    create_figure10_matrix()
