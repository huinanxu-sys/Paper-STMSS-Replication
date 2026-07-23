# Table 5. Ablation of the Size-Velocity Temporal Variance Filter on Non-Biological Debris

Reference: Wind_Debris_Augmented evaluation sequence. The temporal variance filter is the on-line component that suppresses wind-blown debris while preserving biological targets (mosquitoes, fruit flies).

| Configuration | Wind Debris Detected Tracks | Debris Suppression Rate (%) | Culex LocA (%) |
|:---|:---:|:---:|:---:|
| Baseline (MOG2 + CV-KF only) | 685 | 15.8 | 67.15 |
| Proposed (+ Temporal Variance Filter) | 89 | 89.1 | 67.28 |

Adding the temporal variance filter raises the debris suppression rate from 15.8 % to 89.1 % on the Wind_Debris_Aug sequence while preserving the Culex_Transit LocA (67.15 % -> 67.28 %). This isolates the filter as the dominant contribution to non-biological debris rejection.

Script: `evaluation/Table5__VarianceFilterAblation.py`
Data:   `data/csv/table5_ablation_source.csv`