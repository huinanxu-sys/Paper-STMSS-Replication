# Reproduction Guide

Complete guide for reproducing all figures and tables from Paper D17.

---

## Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Optional: Sample videos for Figure 10

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/D17-STMSS-Replication.git
cd D17-STMSS-Replication

# Create virtual environment
python -m venv venv

# Activate environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Figure Reproduction

### Figure 1: Levy Distribution

```bash
cd figures
python plot_figure1_levy_distribution.py
```

**Output**: `Figure1_Levy_Distribution.png/pdf`

**Expected Result**: Comparison of Levy (alpha=1.5, 1.0) vs Gaussian (alpha=2.0) distributions

---

### Figure 2: Monte Carlo Analysis

```bash
cd figures
python plot_figure2_monte_carlo.py
```

**Output**: `Figure2_Monte_Carlo.png/pdf`

**Parameters**:
- Simulations: 1,000,000 (10^6)
- Capture radius: 0.15m
- Max velocity: 2.0 m/s
- Base latency: 18.82ms

**Expected Result**: Capture probability vs latency curve showing 95.2% at 18.82ms

---

### Figure 4: Architecture Diagram

```bash
cd figures
python plot_figure4_architecture.py
```

**Output**: `Figure4_Architecture.png/pdf/svg`

**Expected Result**: STMSS cyber-physical architecture schematic

---

### Figure 6: GA Convergence

```bash
cd figures
python plot_figure6_ga_convergence.py
```

**Input**: `data/csv/figure6_ga_convergence.csv`

**Output**: `Figure6_GA_Convergence.png/pdf/svg`

**Expected Result**: Convergence curve with 10 runs, mean and variance shading

---

### Figure 7: Latency Waterfall

```bash
cd figures
python plot_figure7_latency_waterfall.py
```

**Input**: `data/csv/figure7_latency_data.csv`

**Output**: `Figure7_Latency_Waterfall.png/pdf`

**Expected Result**: Waterfall chart showing STMSS (18.82ms) vs YOLOv8n+ByteTrack (47.29ms)

---

### Figure 8: Survival Probability

```bash
cd figures
python plot_figure8_survival_probability.py
```

**Input**: `data/csv/figure8_survival_probability.csv`

**Output**: `Figure8_Survival_Probability.png/pdf`

**Expected Result**: Survival probability curves with key point at (18.82ms, 95.24%)

---

### Figure 9: PAD Workload

```bash
cd figures
python plot_figure9_pad_workload.py
```

**Input**: `data/csv/figure9_pad_workload.csv`

**Output**: `Figure9_PAD_Workload.png/pdf`

**Expected Result**: Workload reduction chart showing ~49.35% average reduction

---

### Figure 10: Qualitative Trajectories

**Note**: Requires sample videos in `videos/` directory

```bash
cd figures
python Figure10_QualitativeTrajectories__TrackingGrid.py
```

**Output**: `Figure10_Qualitative_Trajectories.png/pdf/svg`

**Expected Result**: 2x2 grid of tracking trajectories

---

## Table Reproduction

### Table 2: Tracking Performance

```bash
cd evaluation
python Table2_TrackingPerformance__STMSS_Runner.py
```

**Input**: `data/ground_truth/*.txt`, sample videos

**Output**: MOT format results in `data/tracker_outputs/STMSS/`

**Metrics**: HOTA, MOTA, IDF1, FP, FN, IDs

---

### Table 3: Ablation Study

```bash
cd evaluation
python Table3_Ablation__VarianceFilter.py
```

**Output**: Ablation study results

---

## Batch Generation

Generate all figures at once:

```bash
cd figures
for script in plot_figure*.py; do
    echo "Running $script..."
    python "$script"
done
python Figure10_QualitativeTrajectories__TrackingGrid.py
```

---

## Verification

### Check Output Files

```bash
# List all generated figures
ls -lh figures/*.png figures/*.pdf

# Verify CSV data
cat data/csv/figure7_latency_data.csv
```

### Expected File Sizes

| File | Expected Size |
|:---|:---|
| Figure1_Levy_Distribution.png | ~200 KB |
| Figure2_Monte_Carlo.png | ~150 KB |
| Figure6_GA_Convergence.png | ~180 KB |
| Figure7_Latency_Waterfall.png | ~120 KB |
| Figure8_Survival_Probability.png | ~100 KB |
| Figure9_PAD_Workload.png | ~90 KB |

---

## Troubleshooting

### Issue: Missing dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Issue: Figure 10 video not found

Figure 10 requires sample videos. Download from [link] or skip this figure.

### Issue: Permission denied (Windows)

Run terminal as Administrator or check file permissions.

---

## Data Consistency Check

Verify that generated outputs match paper values:

```python
# Quick verification script
import pandas as pd

# Check Figure 7 latency data
df = pd.read_csv('data/csv/figure7_latency_data.csv')
assert df[df['System'] == 'STMSS_Proposed']['Total_Latency_ms'].values[0] == 18.82
assert df[df['System'] == 'YOLOv8n_ByteTrack']['Total_Latency_ms'].values[0] == 72.29

print("Data verification passed!")
```

---

## Citation

When using this replication package, please cite:

```bibtex
@article{d17_stmss_2026,
  title={Spatio-Temporal Micro-Swarm Sensing (STMSS): A Cyber-Physical Architecture for High-Speed Target Interception},
  author={[Authors]},
  journal={Computers and Electronics in Agriculture},
  year={2026}
}
```

---

*Last updated: 2026-05-23*
