from ultralytics import YOLO
import cv2
import os

# 1. 加载重型模型 (首次运行会自动下载 yolov8x.pt，约 130MB)
# 注意：如果你之前有自己微调过的权重（比如 best.pt），请把 'yolov8x.pt' 换成你的权重路径！
print("正在加载 YOLOv8x 超大模型...")
model = YOLO('yolov8x.pt') 

# 2. 定义你要跑的真实视频路径
videos = {
    "Aedes_Saccade": "samples/Supp Video 1.mp4",
    "Culex_Transit": "samples/flying_mosquito.mp4",
    # 对于果蝇视频，如果不做截取，CPU跑YOLOv8x可能需要几天几夜！
    "Drosophila_Dense": "samples/drosophila_10.avi" 
}

# 确保输出目录存在
os.makedirs("data/ground_truth", exist_ok=True)

# 3. 开始遍历视频并生成真值
for seq_name, video_path in videos.items():
    print(f"\n=====================================")
    print(f"正在处理视频: {seq_name}")
    print(f"=====================================")
    
    cap = cv2.VideoCapture(video_path)
    gt_path = f"data/ground_truth/{seq_name}.txt"
    
    with open(gt_path, 'w') as f:
        frame_idx = 1
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 使用 ByteTrack 追踪，关闭 verbose 避免终端被刷屏
            # 设置较高的 imgsz (如 1280) 以确保能看清微小目标
            # 设置较低的 conf (如 0.1) 防止漏检微小昆虫
            results = model.track(
                frame, 
                tracker="bytetrack.yaml", 
                persist=True, 
                imgsz=1280,   
                conf=0.1,     
                verbose=False 
            )
            
            # 如果画面中检测到了目标，并且 ByteTrack 分配了 ID
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                for box, track_id, conf in zip(boxes, track_ids, confs):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    
                    # 组装 MOT 1.1 标准格式:
                    # <frame>, <id>, <bb_left>, <bb_top>, <width>, <height>, <conf>, -1, -1, -1
                    line = f"{frame_idx},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
                    f.write(line)
            
            # 打印进度提示
            if frame_idx % 100 == 0:
                print(f"[{seq_name}] 已处理 {frame_idx} 帧...")
                
            frame_idx += 1
            
            # 【重要防御机制】如果你不想把果蝇视频的3.6万帧全跑完，取消下面两行的注释：
            if seq_name == "Drosophila_Dense" and frame_idx > 1000:
                print("果蝇视频已达到 1000 帧截取上限，提前结束！")
                break

    cap.release()
    print(f"✅ {seq_name} 真值生成完毕！文件保存在: {gt_path}")

print("\n所有离线 Ground Truth 生成完毕！可以去跑 TrackEval 了！")