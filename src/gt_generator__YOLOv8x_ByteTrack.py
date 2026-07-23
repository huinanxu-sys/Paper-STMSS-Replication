"""
Offline Ground-Truth Generator (YOLOv8x + ByteTrack).

This helper is used OFFLINE to bootstrap the MOT-format ground-truth
files for the biological target sequences. It is NOT part of the online
STMSS pipeline that ships with the EAAI replication package.

The script expects a GPU host because running YOLOv8x on CPU for the
Drosophila_Dense sequence alone would take several days. The first
invocation downloads yolov8x.pt (~130 MB).

Notes for EAAI reviewers:
    * The MOT 1.1 row format is:
      <frame>, <id>, <bb_left>, <bb_top>, <width>, <height>,
      <conf>, -1, -1, -1
    * All track IDs are positive integers, as required by TrackEval.
    * The script is provided for completeness; the ground-truth files
      in data/ground_truth/ are already produced and committed.
"""

from ultralytics import YOLO
import cv2
import os

# 1. Load the heavy YOLOv8x model (yolov8x.pt is auto-downloaded on first
#    run, approximately 130 MB). Replace 'yolov8x.pt' with a custom weight
#    file (e.g., 'best.pt') if you have fine-tuned weights available.
print("Loading YOLOv8x large model...")
model = YOLO('yolov8x.pt')

# 2. Define the video paths to process
videos = {
    "Aedes_Saccade": "samples/Supp Video 1.mp4",
    "Culex_Transit": "samples/flying_mosquito.mp4",
    # WARNING: Without frame truncation, running YOLOv8x on the
    # Drosophila_Dense sequence on CPU can take several days.
    "Drosophila_Dense": "samples/drosophila_10.avi"
}

# Ensure the output directory exists
os.makedirs("data/ground_truth", exist_ok=True)

# 3. Iterate over the videos and emit ground truth
for seq_name, video_path in videos.items():
    print(f"\n=====================================")
    print(f"Processing video: {seq_name}")
    print(f"=====================================")

    cap = cv2.VideoCapture(video_path)
    gt_path = f"data/ground_truth/{seq_name}.txt"

    with open(gt_path, 'w') as f:
        frame_idx = 1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Use ByteTrack tracking; disable verbose to keep the console clean.
            # Use a large imgsz (e.g. 1280) so that tiny targets are visible.
            # Use a low conf (e.g. 0.1) to avoid missing small insects.
            results = model.track(
                frame,
                tracker="bytetrack.yaml",
                persist=True,
                imgsz=1280,
                conf=0.1,
                verbose=False
            )

            # If the frame contains detections and ByteTrack assigned IDs
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                for box, track_id, conf in zip(boxes, track_ids, confs):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1

                    # Assemble the MOT 1.1 standard format:
                    # <frame>, <id>, <bb_left>, <bb_top>, <width>, <height>,
                    # <conf>, -1, -1, -1
                    line = (
                        f"{frame_idx},{int(track_id)},{x1:.2f},{y1:.2f},"
                        f"{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
                    )
                    f.write(line)

            # Print progress every 100 frames
            if frame_idx % 100 == 0:
                print(f"[{seq_name}] Processed {frame_idx} frames...")

            frame_idx += 1

            # [Important guard] If you do not want to run the full
            # Drosophila_Dense video (~36k frames), uncomment the next two
            # lines to truncate the sequence:
            if seq_name == "Drosophila_Dense" and frame_idx > 1000:
                print("Drosophila_Dense reached the 1000-frame truncation cap.")
                break

    cap.release()
    print(f"[ok] {seq_name} ground truth generated: {gt_path}")

print("\nAll offline ground-truth files are ready for TrackEval evaluation.")
