# ASR-HC & JEV Ensemble: Anatomical Strip-Routed ConvNeXt with Source-Prior Harmonic Calibration

[![WACV 2027](https://img.shields.io/badge/WACV%20Workshop-2027-blue.svg)](https://vap.aau.dk/rws)
[![UPAR Track 1](https://img.shields.io/badge/UPAR%20Challenge-Track%201%20PAR-brightgreen.svg)](https://www.codabench.org/competitions/18180/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official code repository and technical report source for **UPAR Challenge 2027 Track 1 (Pedestrian Attribute Recognition)** at the **Real-World Surveillance: Applications and Challenges Workshop (WACV 2027)**.

---

## 👥 Authors & Team Members
* **Muhammad Abbas**
* **Zaid Bin Naveed**
* **Muhammad Sabeeh**

---

## 🏆 Benchmark Highlights

Evaluated across all **33,407 validation crops** of the harmonized UPAR benchmark (PA-100K, PETA, and Market1501):

| Metric | Baseline GAP | Proposed ASR-HC | JEV Dual-Head Ensemble (70/30) |
| :--- | :---: | :---: | :---: |
| **Harmonic Mean (HM)** $\uparrow$ | 81.33% | 81.05% | **81.86%** *(State-of-the-Art)* |
| **Label mean Accuracy (mA)** | **81.70%** | 78.18% | 80.59% |
| **Instance F1-Score** | 80.97% | **84.15%** | 83.18% |
| **Instance Accuracy** | 70.08% | **74.99%** *(+4.91%)* | 73.25% |
| **Instance Precision** | 77.73% | **86.28%** *(+8.55%)* | 82.74% |

* **Air-Gapped CodaBench Verification:** Verified and evaluated on CodaBench's hidden evaluation test set of **29,248 images** with 0 runtime errors.

---

## 🔬 Core Innovations

```
[Input Pedestrian Crop: 3 x 256 x 192]
                 │
                 ▼
[ConvNeXt-Base Backbone: Stages 0--2 Frozen (VRAM 14.8GB -> 5.6GB)]
                 │
                 ▼
[Stage 4 Trainable Feature Map: C=1024, H' x W']
                 ├───► [Baseline Head: GAP -> Linear(40)] ───────────┐
                 │                                                   │
                 ▼                                                   │
[Anatomical 3-Strip Partitioning: S0 (Head), S1 (Torso), S2 (Lower)] │
                 │                                                   │
                 ▼                                                   │
[8-Head Cross-Strip Multi-Head Attention (CS-MHA) + Pos Embeddings]  │
                 │                                                   │
                 ▼                                                   │
[Disjoint Anatomical Sub-Classifiers: 10 Head, 15 Torso, 15 Lower]  │
                 │                                                   │
                 ▼                                                   │
[Source-Prior Harmonic Calibration + Odds-Ratio Warping]             │
                 │                                                   │
                 ▼                                                   ▼
       [JEV Shared-Stem Dual-Head Ensemble: 0.70 P_base + 0.30 P_asr]
                                     │
                                     ▼
                      [Winning HM: 81.86% (40 Attributes)]
```

### 1. Anatomical Strip Routing (ASR-HC Head)
* Slices deep ConvNeXt Stage-4 feature representations into 3 horizontal anatomical strips:
  * **$S_0$ (Head/Shoulders):** 10 attributes (Age, Gender, Hair length, Hat).
  * **$S_1$ (Torso/Upper-Body):** 15 attributes (Upper clothing length, 12 garment shades).
  * **$S_2$ (Lower-Body/Footwear):** 15 attributes (Lower clothing length, 12 lower shades, Shoes).
* Inter-strip relational reasoning is refined via an **8-head Cross-Strip Attention (CS-MHA)** module with learnable positional embeddings, completely eliminating cross-region feature bleed.

### 2. Asymmetric Source-Prior Harmonic Calibration
* Addresses extreme surveillance class imbalance (e.g., Age-Adult $>93\%$ vs rare clothing colors $<0.5\%$) without test-set leakage:
  $$\tau_j = \begin{cases} \mathrm{clip}\left(0.50 \cdot (\pi_j / 0.50)^\alpha, \; 0.25, \; 0.50\right) & \text{if } \pi_j < 0.50 \\ 0.50 & \text{if } \pi_j \ge 0.50 \end{cases}$$
* **Continuous Odds-Ratio Warping:** Enables custom calibration thresholds inside CodaBench's fixed $>0.5$ evaluation rule:
  $$p^*_j = \frac{p_j / \tau_j}{\frac{p_j}{\tau_j} + \frac{1 - p_j}{1 - \tau_j}}$$
* **Color Competition Suppression:** Margin suppression ($\delta = 0.15$) eliminates multi-color hallucinations on single garments.

### 3. Joint Execution Vectorization (JEV) Dual-Head Engine
* Reuses the frozen backbone features across both Baseline GAP and ASR-HC heads in a **single forward pass**.
* Blends the global sensitivity of GAP with the localized precision of ASR-HC:
  $$P_{\text{ensemble}} = 0.70 \cdot P_{\text{baseline}} + 0.30 \cdot P_{\text{ASR-HC}}$$
  yielding an overall state-of-the-art **81.86% Harmonic Mean**.

---

## 📁 Repository Structure

```text
.
├── src/                               # Core model architecture & calibration modules
│   ├── models_jev_asrhc.py            # ASR-HC Head, CS-MHA, JEV Engine & NativeConvNeXtBase
│   ├── metrics_and_calibration.py     # Asymmetric Calibrator & Continuous Odds-Ratio Warper
│   └── run_pipeline.py                # End-to-end evaluation, inference & CodaBench exporter
├── scripts/
│   └── train_stratified_fast.py       # Domain-stratified FP16 AMP training pipeline
├── report/                            # IEEE/CVF WACV 2027 LaTeX Report Sources
│   ├── main_assignment_3page.tex      # 3-Page Assignment / Technical Report
│   ├── main_workshop.tex              # 6-Page Camera-Ready Workshop Paper
│   ├── main.bib                       # Complete bibliography (16 entries)
│   └── sec/                           # Modular report sections (Intro, Method, Results)
├── presentation/                      # Defense & Keynote Presentation
│   └── genspark_presentation_prompt.md# Comprehensive AI presentation prompt with anti-cliché rules
├── results/
│   └── tables/                        # Official ablation benchmark tables (CSV & Markdown)
├── test_smoke_jev_asrhc.py            # Architectural smoke test suite
├── test_verify_all_3_fixes.py         # Calibration & air-gapped container test suite
├── download_datasets.py               # Dataset preparation script
└── environment.yml                    # Conda environment definition
```

---

## 🚀 Quickstart & Setup

### 1. Environment Installation
```bash
conda env create -f environment.yml
conda activate rws-upar-challenge
```

### 2. Download Datasets
Prepare the harmonized UPAR challenge datasets (PA-100K, PETA, Market1501):
```bash
python download_datasets.py
```

### 3. Verification & Smoke Tests
Run the standalone test suites to verify pure-PyTorch forward passes, gradient flows, and air-gapped sandboxes:
```bash
python test_smoke_jev_asrhc.py
python test_verify_all_3_fixes.py
```

### 4. Running Evaluation Pipeline
To reproduce validation benchmark results on the 33,407 validation crops:
```bash
python src/run_pipeline.py
```

### 5. Packaging CodaBench Code Submission
Generate an air-gapped submission archive containing FP16 model weights, vendored dependencies, and `run.py`:
```bash
python -c "from src.run_pipeline import export_codabench_code_submission; export_codabench_code_submission()"
```

---

## 📊 Comprehensive Ablation Study

| Model Architecture | Calibration Strategy | Harmonic Mean (HM) $\uparrow$ | Mean Accuracy (mA) | Instance F1 | Instance Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Baseline GAP | Standard (0.50) | 81.33% | **81.70%** | 80.97% | 70.08% |
| Baseline GAP | Source-Prior Cal. | 80.84% | 81.54% | 80.14% | 68.97% |
| ASR-HC Head | Standard (0.50) | 80.60% | 77.15% | **84.37%** | **74.99%** |
| ASR-HC Head | Asymmetric Harmonic Cal. | 81.05% | 78.18% | 84.15% | 74.65% |
| **JEV Dual-Head Ensemble** | **Optimal 70/30 Blend** | **81.86%** | **80.59%** | **83.18%** | **73.25%** |

---

## 📖 Citation

If you use this codebase or technical report in your research, please cite:

```bibtex
@inproceedings{abbas2027asrhc,
  title={Anatomical Strip-Routed ConvNeXt with Source-Prior Harmonic Calibration for Pedestrian Attribute Recognition},
  author={Abbas, Muhammad and Bin Naveed, Zaid and Sabeeh, Muhammad},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
  year={2027}
}
```