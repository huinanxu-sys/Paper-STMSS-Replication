# STMSS-Replication

Official replication package for the **STMSS (Spatiotemporal Micro-Sequence Search)**
cyber-physical interception framework, prepared for **Engineering Applications
of Artificial Intelligence (EAAI)** submission.

## What this repository contains

A fully reproducible pipeline that re-creates every quantitative result in the
manuscript from the eight raw test sequences and the offline calibration
data. All scripts are written in pure Python 3.10, depend only on NumPy, SciPy,
OpenCV, filterpy, and matplotlib, and run on a single unaccelerated
Intel Core i5-8250U industrial edge gateway (no GPU, no NPU).

## Hardware environment

- CPU: Intel Core i5-8250U @ 1.6 GHz (locked to performance governor)
- RAM: 8 GB
- OS: Windows 10 / Ubuntu 22.04 (tested on both)
- No discrete GPU, no NPU, no OpenVINO DSP

## Repository layout

```
STMSS-Replication/
|-- data/
|   |-- csv/                          # All quantitative data files (CSVs)
|   `-- ground_truth/                 # MOT-format ground truth annotations
|-- docs/                             # Markdown renderings of the tables
|-- evaluation/                       # Scripts that produce the tables
|-- figures/                          # Scripts + generated outputs (PNG, PDF, SVG)
|-- src/                              # Core STMSS pipeline + YOLOv8x GT generator
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Reproducing the deliverables (one command)

```bash
pip install -r requirements.txt

# Step 1: produce the per-sequence baseline latency measurements
#         (requires 08_Sample_Videos/; the data/csv/table1_baselines.csv
#          file is also committed so this step is optional)
python evaluation/run_baselines_table1.py

# Step 2: build the table deliverables
python evaluation/build_table1.py
python evaluation/build_figure7_data.py
python evaluation/build_figure8_data.py
python evaluation/Table2__StageLatencyBreakdown.py
python evaluation/Table3__StateSpaceBenchmark.py
python evaluation/build_table6_cross_validation.py
python evaluation/Table5__VarianceFilterAblation.py

# Step 3: build the figure deliverables (PNG, PDF, SVG, 600 DPI)
python figures/plot_figure1_levy_distribution.py
python figures/plot_figure2_monte_carlo.py
python figures/plot_figure3_interception_geometry.py
python figures/plot_figure4_architecture.py
python figures/plot_figure6_ga_convergence.py
python figures/plot_figure7_latency_waterfall.py
python figures/plot_figure8_survival_probability.py
python figures/plot_figure9_pad_workload.py
python figures/plot_figure10_qualitative_trajectories.py
python figures/plot_figure11_pareto_front.py
```

All numerical outputs (CSVs, PNGs, PDFs, SVGs) will be regenerated under
`data/csv/`, `docs/`, and `figures/`.

> **Note**: Figure 5 is a flowchart drawn with Visio / PowerPoint and is **not**
> produced by any script in this repository.
>
> **Note**: Figure 10 reads sample videos from a local `08_Sample_Videos/`
> directory; if these are absent, the script still runs and produces the
> trajectory overlays on blank panels. The provided video clips are not
> redistributed in this package for licensing reasons.

## What each table / figure depends on

| Artefact  | Reproducing script                              | Data file                            |
|:----------|:------------------------------------------------|:-------------------------------------|
| Table 1   | `evaluation/build_table1.py`                    | `data/csv/table1_semantic_baselines.csv`, `data/csv/table1_baselines.csv`, `data/csv/table1_metadata.csv` |
| Table 2   | `evaluation/Table2__StageLatencyBreakdown.py`   | `data/csv/table2_stage_latency.csv`  |
| Table 3   | `evaluation/Table3__StateSpaceBenchmark.py`     | `data/csv/table3_state_space.csv`    |
| Table 4   | (raw measurement, no build script)             | `data/csv/table4_tracking_performance.csv` |
| Table 5   | `evaluation/Table5__VarianceFilterAblation.py`  | `data/csv/table5_ablation_source.csv` |
| Table 6   | `evaluation/build_table6_cross_validation.py`   | `data/csv/table6_cross_validation_source.csv` |
| Figure 1  | `figures/plot_figure1_levy_distribution.py`     | (theoretical)                        |
| Figure 2  | `figures/plot_figure2_monte_carlo.py`           | n = 1,000,000 Monte-Carlo samples    |
| Figure 3  | `figures/plot_figure3_interception_geometry.py` | (geometric schematic)                |
| Figure 4  | `figures/plot_figure4_architecture.py`          | (schematic)                          |
| Figure 5  | N/A (flowchart, drawn with Visio / PowerPoint)  | --                                   |
| Figure 6  | `figures/plot_figure6_ga_convergence.py`        | `data/csv/figure6_ga_convergence.csv` |
| Figure 7  | `figures/plot_figure7_latency_waterfall.py`     | `data/csv/figure7_latency_data.csv` (built by `evaluation/build_figure7_data.py`) |
| Figure 8  | `figures/plot_figure8_survival_probability.py`  | `data/csv/figure8_survival_probability.csv` (built by `evaluation/build_figure8_data.py`) |
| Figure 9  | `figures/plot_figure9_pad_workload.py`          | `data/csv/figure9_pad_workload.csv`  |
| Figure 10 | `figures/plot_figure10_qualitative_trajectories.py` | (sample videos)                  |
| Figure 11 | `figures/plot_figure11_pareto_front.py`         | `data/csv/figure11_pareto_front.csv`     |

### Data provenance

All quantitative values in the rendered tables and figures are produced by the listed scripts from the listed raw CSV files. The only authoritative numbers are the per-frame raw measurements in the `data/csv/table1_*.csv` files (and the equivalent source files for the other tables and figures). To regenerate the deliverables from raw data, run the build scripts listed above.

## Headline numbers (from raw CSVs)

- **STMSS T_comp = 18.82 ms** (Culex_Transit, raw mean over 250 frames;
  `data/csv/table1_semantic_baselines.csv`); T_total = T_comp +
  T_mech = 18.82 + 25.0 = **43.82 ms**; safety margin = 75.0 - 43.82 =
  **31.18 ms** against the 75.0 ms aerodynamic deadline.
- **YOLOv8n + ByteTrack** T_comp = 47.29 ms
  (`data/csv/table1_semantic_baselines.csv`); T_total = 72.29 ms ->
  **MARGINAL** (2.71 ms margin).
- **STMSS Debris Rejection = 89.1 %** on the Wind_Debris_Augmented
  sequence (`data/csv/table1_metadata.csv`).
- **MOG2 + Lucas-Kanade** T_comp = 8.37 ms
  (`data/csv/table1_baselines.csv`); T_total = 33.37 ms.
- **MOG2 + SORT** T_comp = 12.87 ms
  (`data/csv/table1_baselines.csv`); T_total = 37.87 ms.
- **MOG2 + IMM (2-Model)** T_comp = 38.14 ms
  (`data/csv/table1_baselines.csv`); T_total = 63.14 ms.
- **Particle Filter (N=500)** violates the 20 ms deadline in **84.7 %**
  of cycles (`data/csv/table3_state_space.csv`).
- **Isolated state-space estimator latency** (Table 3, raw):
  STMSS linear KF = 0.156 ms; IMM = 3.156 ms (20.2x STMSS);
  PF (N=500) = 45.678 ms (292.8x STMSS).
- **Integrated-pipeline speedups** (Table 1, raw Culex_Transit):
  STMSS 18.82 ms vs MOG2_IMM 38.14 ms (2.0x); STMSS vs YOLOv8n
  47.29 ms (2.5x).

## Citation

If you use this code, please cite the manuscript (Engineering Applications
of Artificial Intelligence, under review).

## License

MIT License. See `LICENSE`.
