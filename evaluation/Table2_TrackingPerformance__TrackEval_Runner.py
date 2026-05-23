"""
TrackEval Runner for D17 Paper Assets
Validates Table 2 tracking metrics against ground truth
"""

import os
import shutil
import sys

# Add TrackEval to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TrackEval'))

GT_DIR = os.path.join(os.path.dirname(__file__), '..', '04_Data_GroundTruth')
TRACKER_DIR = os.path.join(os.path.dirname(__file__), '..', '05_Data_TrackerOutputs', 'STMSS')
TRACKEVAL_ROOT = os.path.join(os.path.dirname(__file__), '..', 'TrackEval')

MOT_GT_DIR = os.path.join(TRACKEVAL_ROOT, "data", "gt", "mot_challenge", "BugTracker-train")
MOT_TRACKER_DIR = os.path.join(TRACKEVAL_ROOT, "data", "trackers", "mot_challenge", "BugTracker-train", "STMSS", "data")

print("="*60)
print("D17 Paper Assets - TrackEval Validation")
print("="*60)

# Clean and rebuild
if os.path.exists(MOT_GT_DIR):
    shutil.rmtree(MOT_GT_DIR)
if os.path.exists(MOT_TRACKER_DIR):
    shutil.rmtree(MOT_TRACKER_DIR)
os.makedirs(MOT_TRACKER_DIR, exist_ok=True)

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
    print(f"Mapping: {seq}")
    seq_gt_dir = os.path.join(MOT_GT_DIR, seq, "gt")
    os.makedirs(seq_gt_dir, exist_ok=True)
    
    src_gt = os.path.join(GT_DIR, f"{seq}.txt")
    if os.path.exists(src_gt):
        shutil.copy(src_gt, os.path.join(seq_gt_dir, "gt.txt"))
        # Handle null bytes in GT files
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
        print(f"  WARNING: GT not found: {src_gt}")

    src_tracker = os.path.join(TRACKER_DIR, f"{seq}_STMSS.txt")
    if not os.path.exists(src_tracker):
        src_tracker = os.path.join(TRACKER_DIR, f"{seq}.txt")
    
    if os.path.exists(src_tracker):
        shutil.copy(src_tracker, os.path.join(MOT_TRACKER_DIR, f"{seq}.txt"))
    else:
        print(f"  WARNING: Tracker output not found: {src_tracker}")

print("\nRunning TrackEval...")

# Run TrackEval
from trackeval import Evaluator, datasets, metrics

eval_config = {
    'USE_PARALLEL': False,
    'NUM_PARALLEL_CORES': 1,
    'BREAK_ON_ERROR': True,
    'PRINT_RESULTS': True,
    'PRINT_CONFIG': False,
    'TIME_PROGRESS': False,
    'OUTPUT_SUMMARY': True,
    'OUTPUT_EMPTY_CLASSES': False,
    'OUTPUT_DETAILED': True,
    'PLOT_CURVES': False,
}

dataset_config = {
    'GT_FOLDER': MOT_GT_DIR,
    'TRACKERS_FOLDER': os.path.join(TRACKEVAL_ROOT, "data", "trackers", "mot_challenge", "BugTracker-train"),
    'TRACKERS_TO_EVAL': ['STMSS'],
    'BENCHMARK': 'BugTracker',
    'SPLIT_TO_EVAL': 'train',
    'INPUT_AS_ZIP': False,
    'DO_PREPROC': False,
    'TRACKER_SUB_FOLDER': 'data',
    'OUTPUT_FOLDER': os.path.join(os.path.dirname(__file__), '..', '06_Reports'),
    'SEQ_INFO': {seq: {'seq_length': -1} for seq in sequences},
}

evaluator = Evaluator(eval_config)
dataset_list = [datasets.MotChallenge2DBox(dataset_config)]
metrics_list = [metrics.HOTA(), metrics.CLEAR(), metrics.Identity()]

eval_results = evaluator.evaluate(dataset_list, metrics_list)

print("\n" + "="*60)
print("TrackEval Validation Complete")
print("="*60)
