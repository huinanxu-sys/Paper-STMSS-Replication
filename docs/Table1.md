# Table 1. End-to-End Cyber-Physical Latency and Tracking Benchmarks

Hardware: Intel Core i5-8250U @ 1.6 GHz, 8 GB RAM (CPU only). All measurements on the same 512x512 down-sampled stream.

**What this table reports.** Each row is the *Culex_Transit* deadline-calibration reference row from the raw per-sequence measurement CSVs (`data/csv/table1_semantic_baselines.csv` and `data/csv/table1_baselines.csv`). The other seven sequences are present in those same CSVs for audit but are not aggregated into the headline mean. The headline mean, σ and P95 are the per-frame statistics of the Culex_Transit sequence under the EAAI latency protocol (pre-loaded observations, `gc.disable()`, 100-iteration cache warm-up, locked CPU frequency). The values flow directly from the raw CSVs into the rendered output.

| Pipeline | Hardware | T_comp (ms) | σ (ms) | P95 (ms) | T_mech (ms) | T_total (ms) | T_allow (ms) | Margin (ms) | Debris Rejection | Actuation Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| YOLOv8n + ByteTrack | Pure CPU | 47.29 | 8.20 | 60.96 | 25.0 | 72.29 | 75.0 | 2.71 | N/A (Semantic) | MARGINAL |
| MOG2 + Lucas-Kanade | Pure CPU | 8.37 | 1.53 | 10.51 | 25.0 | 33.37 | 75.0 | 41.63 | 12.4% | GUARANTEED |
| MOG2 + SORT | Pure CPU | 12.87 | 3.61 | 19.32 | 25.0 | 37.87 | 75.0 | 37.13 | 18.7% | GUARANTEED |
| MOG2 + IMM (2-Model) | Pure CPU | 38.14 | 10.34 | 55.42 | 25.0 | 63.14 | 75.0 | 11.86 | 21.0% | MARGINAL |
| STMSS (Proposed) | Pure CPU | 18.82 | 3.21 | 27.44 | 25.0 | 43.82 | 75.0 | 31.18 | 89.1% | GUARANTEED |

Notes: T_allow = 75.0 ms is the deterministic physical upper bound for a 2.0 m/s orthogonal micro-vector penetrating the 15 cm effective capture radius of an industrial air curtain. T_mech = 25.0 ms is the industrial solenoid activation lag (hardware-certified worst-case per the EXAIR Super Air Knife manufacturer datasheet). Debris Rejection Rate is evaluated on the Wind_Debris_Augmented sequence. The P99 worst-case execution time is not stored in this repo; the figure-7 waterfall reads directly from the raw Culex_Transit CSVs.

Script: `evaluation/build_table1.py`
Data:
- `data/csv/table1_semantic_baselines.csv` (YOLOv8n, STMSS per-sequence)
- `data/csv/table1_baselines.csv` (MOG2 raw per-sequence measurements)
- `data/csv/table1_metadata.csv` (hardware + debris rejection)