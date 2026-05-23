# D17 STMSS Replication

Replication package for **"Spatio-Temporal Micro-Swarm Sensing (STMSS): A Cyber-Physical Architecture for High-Speed Target Interception"** (Paper D17)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository provides a complete replication package for the STMSS (Spatio-Temporal Micro-Swarm Sensing) system described in Paper D17. It includes:

- Core algorithm implementations
- Figure generation scripts
- Evaluation tools
- Ground truth data and tracker outputs

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/D17-STMSS-Replication.git
cd D17-STMSS-Replication

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Generate All Figures

```bash
cd figures
python plot_figure1_levy_distribution.py
python plot_figure2_monte_carlo.py
python plot_figure4_architecture.py
python plot_figure6_ga_convergence.py
python plot_figure7_latency_waterfall.py
python plot_figure8_survival_probability.py
python plot_figure9_pad_workload.py
python Figure10_QualitativeTrajectories__TrackingGrid.py
```

### Run Evaluation

```bash
cd evaluation
python Table2_TrackingPerformance__STMSS_Runner.py
```

---

## Repository Structure

```
D17-STMSS-Replication/
├── src/                    # Core algorithm implementations
│   ├── stmss_core__PhotonRestoration_PI-EOKF.py
│   ├── stmss_tracker__FullPipeline.py
│   └── ...
├── figures/                # Figure generation scripts
│   ├── plot_figure1_levy_distribution.py
│   ├── plot_figure2_monte_carlo.py
│   └── ...
├── data/                   # Data files
│   ├── csv/               # CSV data for figures
│   └── ground_truth/      # Ground truth annotations
├── evaluation/             # Evaluation scripts
│   ├── Table2_TrackingPerformance__STMSS_Runner.py
│   └── ...
├── docs/                   # Documentation
├── tests/                  # Unit tests
├── videos/                 # Sample videos (download separately)
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md              # This file
```

---

## Paper Mapping

| Paper Element | Script/Data | Description |
|:---|:---|:---|
| **Figure 1** | `figures/plot_figure1_levy_distribution.py` | Levy vs Gaussian distribution |
| **Figure 2** | `figures/plot_figure2_monte_carlo.py` | Monte Carlo sensitivity analysis (10^6 samples) |
| **Figure 4** | `figures/plot_figure4_architecture.py` | System architecture diagram |
| **Figure 6** | `figures/plot_figure6_ga_convergence.py` | GA convergence curve |
| **Figure 7** | `figures/plot_figure7_latency_waterfall.py` | Latency waterfall chart |
| **Figure 8** | `figures/plot_figure8_survival_probability.py` | Survival probability curves |
| **Figure 9** | `figures/plot_figure9_pad_workload.py` | PAD workload reduction |
| **Figure 10** | `figures/Figure10_QualitativeTrajectories__TrackingGrid.py` | Qualitative trajectories |
| **Table 2** | `evaluation/Table2_TrackingPerformance__STMSS_Runner.py` | Tracking performance metrics |
| **Table 3** | `evaluation/Table3_Ablation__VarianceFilter.py` | Ablation study |

---

## Data Files

### CSV Data (in `data/csv/`)

| File | Used By | Description |
|:---|:---|:---|
| `figure6_ga_convergence.csv` | Figure 6 | GA convergence logs (10 runs) |
| `figure7_latency_data.csv` | Figure 7 | Latency breakdown data |
| `figure8_survival_probability.csv` | Figure 8 | Survival probability data |
| `figure9_pad_workload.csv` | Figure 9 | PAD workload data |

### Ground Truth (in `data/ground_truth/`)

MOT Challenge format annotations for 8 test sequences:
- Aedes_Saccade.txt
- Culex_Transit.txt
- Drosophila_Dense.txt
- levy_test_3objects_alpha05_600f.txt
- levy_test_3objects_alpha10_600f.txt
- levy_test_3objects_alpha15_600f.txt
- synthetic_swarm_stress_test.txt
- wind_debris_augmented.txt

---

## Reproducibility

All figures and tables can be reproduced from the provided scripts and data. See [REPRODUCTION.md](docs/REPRODUCTION.md) for detailed instructions.

### Verification Checklist

- [x] Figure 1: Levy Distribution
- [x] Figure 2: Monte Carlo (10^6 samples)
- [x] Figure 4: Architecture Diagram
- [x] Figure 6: GA Convergence
- [x] Figure 7: Latency Waterfall
- [x] Figure 8: Survival Probability
- [x] Figure 9: PAD Workload
- [x] Figure 10: Qualitative Trajectories
- [x] Table 2: Tracking Performance
- [x] Table 3: Ablation Study

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{d17_stmss_2026,
  title={Spatio-Temporal Micro-Swarm Sensing (STMSS): A Cyber-Physical Architecture for High-Speed Target Interception},
  author={[Authors]},
  journal={Computers and Electronics in Agriculture},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Contact

For questions or issues, please open an issue on GitHub.

---

*Last updated: 2026-05-23*  
*Status: Ready for CEA Submission*
