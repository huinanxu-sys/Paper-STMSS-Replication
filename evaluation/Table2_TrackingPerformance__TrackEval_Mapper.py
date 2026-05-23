import os
import shutil

# 1. 你的干净数据路径
GT_DIR = "data/ground_truth"
TRACKER_DIR = "data/table2_stmss_outputs"

# 2. TrackEval 要求的反人类路径套娃
TRACKEVAL_ROOT = "TrackEval"  # 假设你已经克隆了 TrackEval 仓库
MOT_GT_DIR = os.path.join(TRACKEVAL_ROOT, "data/gt/mot_challenge/BugTracker-train")
MOT_TRACKER_DIR = os.path.join(TRACKEVAL_ROOT, "data/trackers/mot_challenge/BugTracker-train/STMSS/data")

print("========================================")
print("🚀 正在构建 TrackEval 跑分矩阵...")
print("========================================")

# 3. 清理并重建 TrackEval 目录
if os.path.exists(MOT_GT_DIR): shutil.rmtree(MOT_GT_DIR)
if os.path.exists(MOT_TRACKER_DIR): shutil.rmtree(MOT_TRACKER_DIR)
os.makedirs(MOT_TRACKER_DIR, exist_ok=True)

# 4. 把你刚刚洗干净的完美真值全部加进来！
# 包含所有8个视频
sequences = [
    "Culex_Transit",
    "Aedes_Saccade",
    "Drosophila_Dense",
    "levy_test_3objects_alpha05_600f",
    "levy_test_3objects_alpha10_600f",
    "levy_test_3objects_alpha15_600f",
    "synthetic_swarm_stress_test",
    "wind_debris_augmented",
]

for seq in sequences:
    print(f"正在映射序列: {seq}")
    # 构建 GT 目录
    seq_gt_dir = os.path.join(MOT_GT_DIR, seq, "gt")
    os.makedirs(seq_gt_dir, exist_ok=True)
    
    # 拷贝并重命名 GT
    src_gt = os.path.join(GT_DIR, f"{seq}.txt")
    if os.path.exists(src_gt):
        shutil.copy(src_gt, os.path.join(seq_gt_dir, "gt.txt"))
        
        # 智能计算每个视频的实际帧数，生成 seqinfo.ini
        with open(src_gt, 'rb') as f:
            raw = f.read()
        raw = raw.replace(b'\x00', b'')
        text = raw.decode('utf-8', errors='replace')
        lines = text.splitlines()
        frame_ids = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    frame_ids.append(int(line.split(',')[0]))
                except (ValueError, IndexError):
                    continue
        seq_length = max(frame_ids) if frame_ids else 100
            
        with open(os.path.join(MOT_GT_DIR, seq, "seqinfo.ini"), "w") as f:
            f.write(f"[Sequence]\nname={seq}\nseqLength={seq_length}\n") 
    else:
        print(f"❌ 找不到真值文件: {src_gt}")

    # 拷贝 Tracker 结果 (自动兼容是否带 _STMSS 后缀)
    src_tracker = os.path.join(TRACKER_DIR, f"{seq}_STMSS.txt")
    if not os.path.exists(src_tracker):
        src_tracker = os.path.join(TRACKER_DIR, f"{seq}.txt")

    if os.path.exists(src_tracker):
        shutil.copy(src_tracker, os.path.join(MOT_TRACKER_DIR, f"{seq}.txt"))
    else:
        print(f"❌ 找不到追踪结果: {src_tracker}")

print("\n✅ 数据映射完成！准备呼叫 TrackEval...")
print("👉 接下来，请在终端运行以下官方命令提取你的 Table 2 数据：")
print("python TrackEval/scripts/run_mot_challenge.py --BENCHMARK BugTracker --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL STMSS --METRICS HOTA CLEAR Identity --DO_PREPROC False")