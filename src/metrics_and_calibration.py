"""
metrics_and_calibration.py
==========================
Official UPAR Challenge (WACV 2027) Track 1 Evaluation Metrics &
Source-Prior Harmonic Calibrator (ASR-HC).

Components:
-----------
1. `compute_upar_metrics`:
   Official Track 1 evaluation protocol:
   - Label-based mean Accuracy (mA)
   - Instance-based Precision, Recall, Accuracy, and F1 score (Ins-F1)
   - Official Primary Ranking Score: Harmonic Mean of mA and Ins-F1:
       HM = 2 * (mA * Ins-F1) / (mA + Ins-F1)

2. `SourcePriorHarmonicCalibrator`:
   - Uses strictly training-split positive priors pi_j = P(y_j = 1 | Train).
   - Zero Cross-Split Leakage: Never inspects validation or test labels.
   - Harmonic Threshold Calibration:
     Bayesian logit adjustment dynamically recalibrates decision boundaries
     to counteract extreme attribute imbalance (e.g. rare colors vs dominant adult).
   - Intra-Group Color Competition:
     Implements competitive margin suppression across the 12 UpperBody and 12 LowerBody
     color attributes to prevent multi-color hallucinations on single pedestrians,
     harmonizing label accuracy with instance-level precision.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# Canonical 40-attribute list matching UPAR Track 1 ground truth specification
UPAR_40_ATTRIBUTES: Tuple[str, ...] = (
    "Age-Young",                        # 0
    "Age-Adult",                        # 1
    "Age-Old",                          # 2
    "Gender-Female",                    # 3
    "Hair-Length-Short",                # 4
    "Hair-Length-Long",                 # 5
    "Hair-Length-Bald",                 # 6
    "UpperBody-Length-Short",           # 7
    "UpperBody-Color-Black",            # 8
    "UpperBody-Color-Blue",             # 9
    "UpperBody-Color-Brown",            # 10
    "UpperBody-Color-Green",            # 11
    "UpperBody-Color-Grey",             # 12
    "UpperBody-Color-Orange",           # 13
    "UpperBody-Color-Pink",             # 14
    "UpperBody-Color-Purple",           # 15
    "UpperBody-Color-Red",              # 16
    "UpperBody-Color-White",            # 17
    "UpperBody-Color-Yellow",           # 18
    "UpperBody-Color-Other",            # 19
    "LowerBody-Length-Short",           # 20
    "LowerBody-Color-Black",            # 21
    "LowerBody-Color-Blue",             # 22
    "LowerBody-Color-Brown",            # 23
    "LowerBody-Color-Green",            # 24
    "LowerBody-Color-Grey",             # 25
    "LowerBody-Color-Orange",           # 26
    "LowerBody-Color-Pink",             # 27
    "LowerBody-Color-Purple",           # 28
    "LowerBody-Color-Red",              # 29
    "LowerBody-Color-White",            # 30
    "LowerBody-Color-Yellow",           # 31
    "LowerBody-Color-Other",            # 32
    "LowerBody-Type-Trousers&Shorts",   # 33
    "LowerBody-Type-Skirt&Dress",       # 34
    "Accessory-Backpack",               # 35
    "Accessory-Bag",                    # 36
    "Accessory-Glasses-Normal",         # 37
    "Accessory-Glasses-Sun",            # 38
    "Accessory-Hat",                    # 39
)

# Mutually competitive groups
UPPER_COLOR_INDICES: List[int] = list(range(8, 20))  # 12 UpperBody colors
LOWER_COLOR_INDICES: List[int] = list(range(21, 33)) # 12 LowerBody colors
AGE_GROUP_INDICES: List[int] = [0, 1, 2]             # Young, Adult, Old
HAIR_GROUP_INDICES: List[int] = [4, 5, 6]            # Short, Long, Bald
LOWER_TYPE_INDICES: List[int] = [33, 34]             # Trousers&Shorts, Skirt&Dress


# ==============================================================================
# OFFICIAL UPAR TRACK 1 METRICS EVALUATION
# ==============================================================================

def compute_upar_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: Union[float, np.ndarray, torch.Tensor] = 0.5,
    attribute_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    r"""
    Computes the official UPAR Challenge Track 1 metrics.
    
    Metrics Computed:
    1. Label-based mean Accuracy (mA):
       For each attribute j:
           TPR_j = TP_j / P_j
           TNR_j = TN_j / N_j
           Acc_j = 0.5 * (TPR_j + TNR_j)
       mA = mean(Acc_j) across all 40 attributes.
       
    2. Instance-based metrics:
       For each pedestrian instance i:
           Intersection = |Y_i cap \hat{Y}_i|
           Union        = |Y_i cup \hat{Y}_i|
           Precision_i  = Intersection / |\hat{Y}_i|
           Recall_i     = Intersection / |Y_i|
           F1_i         = 2 * Precision_i * Recall_i / (Precision_i + Recall_i)
           Accuracy_i   = Intersection / Union
       Ins-F1, Ins-Acc, Ins-Prec, Ins-Rec = mean across all instances.
       
    3. Harmonic Mean (HM) [Official Challenge Ranking Criterion]:
       HM = 2 * (mA * Ins-F1) / (mA + Ins-F1)
       
    Args:
        y_true: Binary ground-truth matrix of shape (N, L)
        y_pred: Predicted probabilities or binary predictions of shape (N, L)
        threshold: Decision threshold (scalar or vector of shape (L,))
        attribute_names: Optional list of attribute names (defaults to UPAR_40_ATTRIBUTES)
        
    Returns:
        Dict containing mA, instance_f1, harmonic_mean, instance_acc,
        instance_prec, instance_rec, per_attribute_acc, and formatted summary string.
    """
    # Convert PyTorch tensors to NumPy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.detach().cpu().numpy()
        
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    assert y_true.shape == y_pred.shape, (
        f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
    )
    
    N, L = y_true.shape
    if attribute_names is None:
        attribute_names = UPAR_40_ATTRIBUTES if L == 40 else [f"Attr_{j}" for j in range(L)]
        
    # Binarize predictions using threshold(s)
    if isinstance(threshold, (int, float)):
        y_bin = (y_pred >= threshold).astype(np.int32)
    else:
        thresh_arr = np.asarray(threshold, dtype=np.float64)
        if thresh_arr.ndim == 1 and len(thresh_arr) == L:
            y_bin = (y_pred >= thresh_arr[None, :]).astype(np.int32)
        else:
            y_bin = (y_pred >= thresh_arr).astype(np.int32)
            
    # --------------------------------------------------------------------------
    # 1. Label-based Evaluation (mA)
    # --------------------------------------------------------------------------
    per_attribute_acc = np.zeros(L, dtype=np.float64)
    per_attribute_tpr = np.zeros(L, dtype=np.float64)
    per_attribute_tnr = np.zeros(L, dtype=np.float64)
    
    for j in range(L):
        yt = y_true[:, j]
        yp = y_bin[:, j]
        
        P = np.sum(yt == 1)
        N_neg = np.sum(yt == 0)
        
        TP = np.sum((yt == 1) & (yp == 1))
        TN = np.sum((yt == 0) & (yp == 0))
        
        tpr = (TP / P) if P > 0 else 1.0
        tnr = (TN / N_neg) if N_neg > 0 else 1.0
        
        acc = 0.5 * (tpr + tnr)
        per_attribute_acc[j] = acc
        per_attribute_tpr[j] = tpr
        per_attribute_tnr[j] = tnr
        
    mA = float(np.mean(per_attribute_acc))
    
    # --------------------------------------------------------------------------
    # 2. Instance-based Evaluation (Ins-Acc, Ins-Prec, Ins-Rec, Ins-F1)
    # --------------------------------------------------------------------------
    intersection = np.sum(y_true & y_bin, axis=1).astype(np.float64)
    union = np.sum(y_true | y_bin, axis=1).astype(np.float64)
    pred_sum = np.sum(y_bin, axis=1).astype(np.float64)
    true_sum = np.sum(y_true, axis=1).astype(np.float64)
    
    # Per-instance precision, recall, accuracy
    ins_prec = np.zeros(N, dtype=np.float64)
    ins_rec = np.zeros(N, dtype=np.float64)
    ins_acc = np.zeros(N, dtype=np.float64)
    ins_f1 = np.zeros(N, dtype=np.float64)
    
    for i in range(N):
        p_count = pred_sum[i]
        t_count = true_sum[i]
        u_count = union[i]
        inter = intersection[i]
        
        # Instance precision
        prec_i = (inter / p_count) if p_count > 0 else (1.0 if t_count == 0 else 0.0)
        # Instance recall
        rec_i = (inter / t_count) if t_count > 0 else (1.0 if p_count == 0 else 0.0)
        # Instance accuracy
        acc_i = (inter / u_count) if u_count > 0 else 1.0
        # Instance F1
        denom = prec_i + rec_i
        f1_i = (2.0 * prec_i * rec_i / denom) if denom > 0 else 0.0
        
        ins_prec[i] = prec_i
        ins_rec[i] = rec_i
        ins_acc[i] = acc_i
        ins_f1[i] = f1_i
        
    mean_ins_prec = float(np.mean(ins_prec))
    mean_ins_rec = float(np.mean(ins_rec))
    mean_ins_acc = float(np.mean(ins_acc))
    mean_ins_f1 = float(np.mean(ins_f1))
    
    # --------------------------------------------------------------------------
    # 3. Official Harmonic Mean (HM) Ranking Criterion
    # --------------------------------------------------------------------------
    denom_hm = mA + mean_ins_f1
    harmonic_mean = (2.0 * mA * mean_ins_f1 / denom_hm) if denom_hm > 0 else 0.0
    
    # Formatted summary string
    summary = (
        f"UPAR Track 1 Evaluation Summary:\n"
        f"  * Harmonic Mean (HM)   : {harmonic_mean * 100:.2f}%\n"
        f"  * Label-based mA       : {mA * 100:.2f}%\n"
        f"  * Instance-based F1    : {mean_ins_f1 * 100:.2f}%\n"
        f"  * Instance-based Acc   : {mean_ins_acc * 100:.2f}%\n"
        f"  * Instance-based Prec  : {mean_ins_prec * 100:.2f}%\n"
        f"  * Instance-based Rec   : {mean_ins_rec * 100:.2f}%\n"
        f"  * Evaluated Attributes : {L} across {N} instances"
    )
    
    return {
        "harmonic_mean": harmonic_mean,
        "mA": mA,
        "instance_f1": mean_ins_f1,
        "instance_acc": mean_ins_acc,
        "instance_prec": mean_ins_prec,
        "instance_rec": mean_ins_rec,
        "per_attribute_acc": per_attribute_acc,
        "per_attribute_tpr": per_attribute_tpr,
        "per_attribute_tnr": per_attribute_tnr,
        "attribute_names": list(attribute_names),
        "summary": summary,
    }


# ==============================================================================
# SOURCE-PRIOR HARMONIC CALIBRATOR (ASR-HC)
# ==============================================================================

class SourcePriorHarmonicCalibrator:
    """
    Source-Prior Harmonic Calibrator with Intra-Group Color Competition.
    
    Theoretical Formulation:
    -------------------------
    1. Zero Cross-Split Leakage Guarantee:
       Calibrates decision boundaries exclusively using training-split positive
       priors pi_j = P(y_j = 1 | Train). Validation/test labels are strictly excluded.
       
    2. Harmonic Prior Threshold Calibration:
       For imbalanced attributes, default 0.5 threshold heavily penalizes rare
       attributes (low TPR -> collapsed mA). Bayesian logit adjustment calculates:
           tau_j = sigmoid(logit(0.5) - alpha * log((pi_j + eps) / (1 - pi_j + eps)))
       clipped to [min_thresh, max_thresh].
       Rare attributes receive calibrated lower thresholds (increasing sensitivity),
       while common attributes remain stable, optimizing the Harmonic Mean.
       
    3. Intra-Group Color Competition:
       In pedestrian surveillance, color attributes (12 UpperBody colors, 12 LowerBody
       colors) represent competing mutually exclusive or dominant shades.
       Standard independent sigmoids often predict conflicting combinations
       (e.g., Black, Blue, and Grey simultaneously).
       The calibrator enforces competitive margin suppression:
           If top color c*(1) has confidence p*(1), any runner-up color c_k with
           margin p*(1) - p_k > delta is competitively suppressed below threshold.
    """
    def __init__(
        self,
        priors: Optional[Union[Dict[str, float], Sequence[float], np.ndarray]] = None,
        alpha: float = 0.30,
        color_margin: float = 0.15,
        min_thresh: float = 0.25,
        max_thresh: float = 0.55,
        eps: float = 1e-6,
    ) -> None:
        self.alpha = alpha
        self.color_margin = color_margin
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.eps = eps
        
        self.attribute_names = list(UPAR_40_ATTRIBUTES)
        self.priors: Optional[np.ndarray] = None
        self.calibrated_thresholds: Optional[np.ndarray] = None
        
        if priors is not None:
            self.set_priors(priors)

    def fit_from_training_data(
        self,
        y_train_true: Union[np.ndarray, torch.Tensor],
        attribute_names: Optional[Sequence[str]] = None,
    ) -> SourcePriorHarmonicCalibrator:
        """
        Calculates positive training priors pi_j = mean(y_train_true[:, j])
        from the training split ground truth matrix.
        """
        if isinstance(y_train_true, torch.Tensor):
            y_train_true = y_train_true.detach().cpu().numpy()
        y_train_true = np.asarray(y_train_true, dtype=np.float64)
        
        if attribute_names is not None:
            self.attribute_names = list(attribute_names)
            
        priors = np.mean(y_train_true, axis=0)
        self.set_priors(priors)
        return self

    def set_priors(
        self,
        priors: Union[Dict[str, float], Sequence[float], np.ndarray],
    ) -> None:
        """
        Explicitly sets training split positive priors.
        Accepts dict {attribute_name: prior_val} or sequence/array of length 40.
        """
        if isinstance(priors, dict):
            priors_list = [priors.get(attr, 0.05) for attr in self.attribute_names]
            self.priors = np.clip(np.array(priors_list, dtype=np.float64), self.eps, 1.0 - self.eps)
        else:
            self.priors = np.clip(np.asarray(priors, dtype=np.float64), self.eps, 1.0 - self.eps)
            
        self._compute_calibrated_thresholds()

    def _compute_calibrated_thresholds(self) -> None:
        """
        Computes the calibrated decision threshold vector tau in [min_thresh, max_thresh]^40.
        Uses asymmetric harmonic prior calibration:
          - For rare attributes (pi_j < 0.5):
              tau_j = 0.5 * ((pi_j / 0.5) ** alpha) clipped to [min_thresh, 0.50]
            recovering minority-class recall and boosting label mA.
          - For dominant attributes (pi_j >= 0.5):
              tau_j is anchored near 0.50 (bounded by max_thresh)
            preventing precision erosion on dominant demographic classes and preserving Instance F1.
        """
        if self.priors is None:
            raise ValueError("Training priors have not been initialized. Call fit_from_training_data or set_priors.")
            
        pi = self.priors
        tau = np.full_like(pi, 0.50, dtype=np.float64)
        rare_mask = pi < 0.50
        tau[rare_mask] = 0.50 * ((pi[rare_mask] / 0.50) ** self.alpha)
        
        # Bound thresholds: rare attributes lower bounded by min_thresh (default 0.25),
        # dominant attributes upper bounded by max_thresh (default 0.55).
        self.calibrated_thresholds = np.clip(tau, self.min_thresh, self.max_thresh)

    def optimize_alpha_on_train(
        self,
        y_train_true: Union[np.ndarray, torch.Tensor],
        y_train_prob: Union[np.ndarray, torch.Tensor],
        alpha_candidates: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6),
    ) -> float:
        """
        Optimizes alpha strictly on training split data to maximize the training Harmonic Mean.
        Guarantees zero validation/test leakage.
        """
        best_alpha = self.alpha
        best_hm = -1.0
        
        # Ensure priors are fitted from train
        self.fit_from_training_data(y_train_true)
        
        for cand in alpha_candidates:
            self.alpha = cand
            self._compute_calibrated_thresholds()
            
            calib_preds = self.calibrate(y_train_prob, apply_competition=True)
            metrics = compute_upar_metrics(y_train_true, calib_preds, threshold=0.5)
            hm = metrics["harmonic_mean"]
            
            if hm > best_hm:
                best_hm = hm
                best_alpha = cand
                
        self.alpha = best_alpha
        self._compute_calibrated_thresholds()
        return best_alpha

    def apply_color_competition(
        self,
        probs: np.ndarray,
        group_indices: List[int],
        margin: Optional[float] = None,
        confidence_floor: float = 0.20,
    ) -> np.ndarray:
        """
        Applies competitive margin suppression within an attribute group (e.g. 12 colors).
        
        If the dominant color c*(1) has probability p*(1) >= confidence_floor:
        Any rival color in the group with p*(1) - p_k > margin is suppressed.
        """
        margin_val = self.color_margin if margin is None else margin
        adjusted_probs = probs.copy()
        group_slice = adjusted_probs[:, group_indices]  # (N, num_group_attrs)
        
        # Top color probability per instance
        top_prob = np.max(group_slice, axis=1, keepdims=True)  # (N, 1)
        
        # Mask where top color is confident enough to trigger competition
        competitive_mask = (top_prob >= confidence_floor)
        
        # Rival colors below the margin boundary
        suppress_mask = competitive_mask & ((top_prob - group_slice) > margin_val)
        
        # Suppress rival probabilities below decision threshold
        group_slice[suppress_mask] = np.minimum(group_slice[suppress_mask] * 0.5, 0.05)
        adjusted_probs[:, group_indices] = group_slice
        
        return adjusted_probs

    def calibrate_probabilities(
        self,
        probs: Union[np.ndarray, torch.Tensor],
        apply_competition: bool = True,
    ) -> np.ndarray:
        """
        Applies intra-group color competition on raw model probabilities.
        """
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        probs = np.asarray(probs, dtype=np.float64).copy()
        
        if apply_competition:
            # 1. UpperBody 12-color competition
            probs = self.apply_color_competition(probs, UPPER_COLOR_INDICES)
            # 2. LowerBody 12-color competition
            probs = self.apply_color_competition(probs, LOWER_COLOR_INDICES)
            # 3. Hair-Length 3-way competition (Short, Long, Bald)
            probs = self.apply_color_competition(probs, HAIR_GROUP_INDICES, margin=0.20)
            
        return probs

    def calibrate(
        self,
        probs: Union[np.ndarray, torch.Tensor],
        apply_competition: bool = True,
    ) -> np.ndarray:
        """
        Full calibration pipeline:
        1. Applies intra-group color and morphology competition.
        2. Applies source-prior harmonic decision thresholds tau.
        
        Returns:
            binary_predictions: Array in {0, 1}^(N x 40)
        """
        if self.calibrated_thresholds is None:
            self._compute_calibrated_thresholds()
            
        # Competition-adjusted probabilities
        adj_probs = self.calibrate_probabilities(probs, apply_competition=apply_competition)
        
        # Thresholding against calibrated per-attribute decision boundaries
        binary_preds = (adj_probs >= self.calibrated_thresholds[None, :]).astype(np.int32)
        return binary_preds

    def calibrate_continuous_for_05(
        self,
        probs: Union[np.ndarray, torch.Tensor],
        apply_competition: bool = True,
    ) -> np.ndarray:
        """
        Continuous Probability Warping for CodaBench's fixed '> 0.5' decision threshold.
        
        Applies intra-group color competition, then transforms probabilities relative
        to the calibrated decision threshold vector tau using odds-ratio warping:
            warped_p = (p / tau) / ((p / tau) + ((1 - p) / (1 - tau)))
            
        Mathematical Guarantee:
            (warped_p >= 0.5) <==> (p >= tau)
        
        This allows CodaBench and downstream benchmark evaluators that hardcode
        a threshold of 0.5 to receive smooth, continuous probabilities whose
        decision boundary at 0.5 exactly matches our optimal harmonic threshold tau_j.
        """
        if self.calibrated_thresholds is None:
            self._compute_calibrated_thresholds()
            
        # 1. Apply intra-group competition
        p = self.calibrate_probabilities(probs, apply_competition=apply_competition)
        
        # 2. Numerical safety clip
        eps = 1e-7
        p = np.clip(p, eps, 1.0 - eps)
        tau = np.clip(self.calibrated_thresholds[None, :], eps, 1.0 - eps)
        
        # 3. Odds-ratio probability warping
        odds_p = p / tau
        odds_comp = (1.0 - p) / (1.0 - tau)
        warped_p = odds_p / (odds_p + odds_comp)
        
        return np.clip(warped_p, 0.0, 1.0)


    def evaluate_and_compare(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_prob: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Evaluates and compares uncalibrated (threshold 0.5) vs calibrated performance.
        Returns comparative metrics showing harmonic gain.
        """
        raw_metrics = compute_upar_metrics(y_true, y_prob, threshold=0.5)
        calib_preds = self.calibrate(y_prob, apply_competition=True)
        calib_metrics = compute_upar_metrics(y_true, calib_preds, threshold=0.5)
        
        hm_gain = (calib_metrics["harmonic_mean"] - raw_metrics["harmonic_mean"]) * 100
        ma_gain = (calib_metrics["mA"] - raw_metrics["mA"]) * 100
        f1_gain = (calib_metrics["instance_f1"] - raw_metrics["instance_f1"]) * 100
        
        report = (
            f"=== Source-Prior Harmonic Calibration Impact ===\n"
            f"  Metric              | Baseline (0.5) | Calibrated (ASR-HC) | Gain\n"
            f"  --------------------+----------------+---------------------+--------\n"
            f"  Harmonic Mean (HM)  | {raw_metrics['harmonic_mean']*100:13.2f}% | {calib_metrics['harmonic_mean']*100:18.2f}% | {hm_gain:+5.2f}%\n"
            f"  Label-based mA      | {raw_metrics['mA']*100:13.2f}% | {calib_metrics['mA']*100:18.2f}% | {ma_gain:+5.2f}%\n"
            f"  Instance-based F1   | {raw_metrics['instance_f1']*100:13.2f}% | {calib_metrics['instance_f1']*100:18.2f}% | {f1_gain:+5.2f}%\n"
        )
        
        return {
            "baseline_metrics": raw_metrics,
            "calibrated_metrics": calib_metrics,
            "gain_hm": hm_gain,
            "gain_ma": ma_gain,
            "gain_f1": f1_gain,
            "report": report,
        }
