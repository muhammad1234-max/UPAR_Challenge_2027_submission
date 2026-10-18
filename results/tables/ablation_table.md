# UPAR Challenge Track 1: ASR-HC Ablation Study Results

| Architecture Head      | Calibration Strategy       |   Harmonic Mean (HM %) |   Label mA (%) |   Instance F1 (%) |   Instance Acc (%) |   Instance Prec (%) |   Instance Rec (%) |
|:-----------------------|:---------------------------|-----------------------:|---------------:|------------------:|-------------------:|--------------------:|-------------------:|
| Baseline GAP           | Standard (0.5)             |                  81.33 |          81.70 |             80.97 |              70.08 |               77.73 |              85.81 |
| Baseline GAP           | Source-Prior Calibrated    |                  80.84 |          81.54 |             80.14 |              68.97 |               75.88 |              86.08 |
| ASR-HC Head            | Standard (0.5)             |                  80.60 |          77.15 |             84.37 |              74.99 |               84.29 |              85.21 |
| ASR-HC Head            | Source-Prior Calibrated    |                  81.05 |          78.18 |             84.15 |              74.65 |               83.19 |              85.88 |
| JEV Dual-Head Ensemble | Optimal Blend (0.70/0.30)  |                  81.86 |          80.59 |             83.18 |              73.25 |               81.47 |              85.90 |

