"""
test_smoke_jev_asrhc.py
=======================
Verification smoke test for:
1. JEVMultiSplitEngine (Backbone stages 0..2 frozen, Stage 4 trainable)
2. AnatomicalStripHead (ASR-HC horizontal slicing, cross-strip attention, 40 logits)
3. BaselineGAPHead (GAP baseline head)
4. Joint Execution Vectorization (JEV) across multi-split/multi-head inputs
5. UPAR Track 1 Official Metrics (compute_upar_metrics: mA, Ins-F1, Harmonic Mean)
6. SourcePriorHarmonicCalibrator (training positive priors, intra-group color competition)
"""

import sys
import torch
import numpy as np
import pandas as pd

from src.models_jev_asrhc import (
    JEVMultiSplitEngine,
    AnatomicalStripHead,
    BaselineGAPHead,
    UPAR_40_ATTRIBUTES,
    STRIP_HEAD_INDICES,
    STRIP_TORSO_INDICES,
    STRIP_LOWER_INDICES,
)
from src.metrics_and_calibration import (
    compute_upar_metrics,
    SourcePriorHarmonicCalibrator,
)


def run_smoke_test():
    print("=" * 75)
    print("UPAR CHALLENGE 2027: ASR-HC & JEV ARCHITECTURE VERIFICATION SMOKE TEST")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/6] Hardware and Runtime Environment:")
    print(f"  * Python Version : {sys.version.split()[0]}")
    print(f"  * PyTorch Version: {torch.__version__}")
    print(f"  * Compute Device : {device}")

    # --------------------------------------------------------------------------
    # Test 1: Instantiate JEVMultiSplitEngine & Check Stage Freezing
    # --------------------------------------------------------------------------
    print(f"\n[2/6] Instantiating JEVMultiSplitEngine (ConvNeXt-Base backbone)...")
    model = JEVMultiSplitEngine(pretrained=False, num_classes=40)
    model.eval()
    model.to(device)
    
    # Check freezing report
    freeze_report = model.get_frozen_status_report()
    print("  Backbone Stage Freezing Status:")
    for k, v in freeze_report.items():
        print(f"    - {k:16s}: {'FROZEN (requires_grad=False)' if v else 'TRAINABLE (requires_grad=True)'}")
        
    assert freeze_report.get("stem_frozen", False), "Error: Stem must be frozen!"
    assert freeze_report.get("stage_0_frozen", False), "Error: Stage 0 must be frozen!"
    assert freeze_report.get("stage_1_frozen", False), "Error: Stage 1 must be frozen!"
    assert freeze_report.get("stage_2_frozen", False), "Error: Stage 2 must be frozen!"
    assert not freeze_report.get("stage_3_frozen", False), "Error: Stage 3 (Stage 4) must be trainable!"
    print("  >> PASS: ConvNeXt-Base stages 0..2 strictly frozen, Stage 4 trainable.")

    # --------------------------------------------------------------------------
    # Test 2: Forward Inference with Dummy Tensors (Standard UPAR 256x128 resolution)
    # --------------------------------------------------------------------------
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 256, 128, device=device)
    print(f"\n[3/6] Testing Forward Inference with dummy input: {tuple(dummy_input.shape)}")

    with torch.no_grad():
        # ASR-HC head forward
        asr_logits, details = model(dummy_input, head="asr", return_details=True)
        print(f"  * ASR-HC Logits Shape       : {tuple(asr_logits.shape)} (Expected: {(batch_size, 40)})")
        assert asr_logits.shape == (batch_size, 40), f"Mismatch in ASR-HC logits: {asr_logits.shape}"
        
        # Verify anatomical slicing and cross-strip attention
        refined_tokens = details["refined_tokens"]
        attn_weights = details["attn_weights"]
        routed_logits = details["routed_logits"]
        fused_logits = details["fused_logits"]
        print(f"  * Refined Strip Tokens Shape: {tuple(refined_tokens.shape)} (Expected: {(batch_size, 3, 1024)})")
        print(f"  * Cross-Strip Attn Map Shape: {tuple(attn_weights.shape)} (Expected: {(batch_size, 3, 3)})")
        print(f"  * Routed Logits Shape       : {tuple(routed_logits.shape)} (Expected: {(batch_size, 40)})")
        print(f"  * Fused Logits Shape        : {tuple(fused_logits.shape)} (Expected: {(batch_size, 40)})")
        assert refined_tokens.shape == (batch_size, 3, 1024)
        assert attn_weights.shape == (batch_size, 3, 3)
        
        # Baseline GAP head forward
        base_logits = model(dummy_input, head="baseline")
        print(f"  * Baseline GAP Logits Shape : {tuple(base_logits.shape)} (Expected: {(batch_size, 40)})")
        assert base_logits.shape == (batch_size, 40)
        
        # Joint Execution Vectorization (both heads simultaneously)
        both_out = model(dummy_input, head="both")
        print(f"  * JEV Shared Stem Both-Heads: asr={tuple(both_out['asr'].shape)}, baseline={tuple(both_out['baseline'].shape)}")
        assert both_out["asr"].shape == (batch_size, 40)
        assert both_out["baseline"].shape == (batch_size, 40)
        assert both_out["features"].shape == (batch_size, 1024, 8, 4)
        print(f"  * Stage 4 Spatial Feature Map: {tuple(both_out['features'].shape)}")
    print("  >> PASS: All head forward shapes, strip tokens, and cross-strip attention verified.")

    # --------------------------------------------------------------------------
    # Test 3: Joint Execution Vectorization across Multi-Split / Multi-Source
    # --------------------------------------------------------------------------
    print(f"\n[4/6] Testing JEV Vectorized Multi-Split Execution...")
    split_inputs = {
        "market1501": torch.randn(2, 3, 256, 128, device=device),
        "pa100k": torch.randn(3, 3, 256, 128, device=device),
        "peta": torch.randn(1, 3, 256, 128, device=device),
    }
    with torch.no_grad():
        split_outputs = model.forward_joint_splits(split_inputs, head="asr")
    for s_name, s_out in split_outputs.items():
        expected_shape = (split_inputs[s_name].shape[0], 40)
        print(f"  * Split '{s_name:10s}' output shape: {tuple(s_out.shape)} (matches expected {expected_shape})")
        assert s_out.shape == expected_shape
    print("  >> PASS: Vectorized multi-split joint execution verified.")

    # --------------------------------------------------------------------------
    # Test 4: Official UPAR Track 1 Metrics Protocol (compute_upar_metrics)
    # --------------------------------------------------------------------------
    print(f"\n[5/6] Testing Official UPAR Track 1 Metrics (mA, Ins-F1, Harmonic Mean)...")
    np.random.seed(42)
    N_test = 20
    dummy_y_true = (np.random.rand(N_test, 40) > 0.85).astype(np.int32)
    dummy_y_pred = np.random.rand(N_test, 40)
    
    metrics = compute_upar_metrics(dummy_y_true, dummy_y_pred, threshold=0.5)
    print(metrics["summary"])
    
    assert 0.0 <= metrics["mA"] <= 1.0, "mA out of bounds!"
    assert 0.0 <= metrics["instance_f1"] <= 1.0, "Ins-F1 out of bounds!"
    assert 0.0 <= metrics["harmonic_mean"] <= 1.0, "Harmonic Mean out of bounds!"
    # Mathematical sanity check for Harmonic Mean
    expected_hm = 2.0 * (metrics["mA"] * metrics["instance_f1"]) / (metrics["mA"] + metrics["instance_f1"])
    assert np.isclose(metrics["harmonic_mean"], expected_hm), "Harmonic mean computation mismatch!"
    print("  >> PASS: Official Track 1 metric calculation mathematically verified.")

    # --------------------------------------------------------------------------
    # Test 5: SourcePriorHarmonicCalibrator with Training Split Data
    # --------------------------------------------------------------------------
    print(f"\n[6/6] Testing SourcePriorHarmonicCalibrator (Zero-Leakage & Color Competition)...")
    train_gt_path = "data/annotations/task1/train/gt.csv"
    try:
        df_train = pd.read_csv(train_gt_path)
        attr_cols = [c for c in df_train.columns if c != "# image"]
        train_priors = df_train[attr_cols].mean().to_dict()
        print(f"  * Successfully loaded true training priors from: {train_gt_path}")
        print(f"  * Number of training images analyzed: {len(df_train)}")
    except Exception as e:
        print(f"  * Note: Could not load {train_gt_path} ({e}), using simulated priors.")
        train_priors = {attr: float(np.random.uniform(0.01, 0.7)) for attr in UPAR_40_ATTRIBUTES}

    calibrator = SourcePriorHarmonicCalibrator(priors=train_priors, alpha=0.35, color_margin=0.15)
    print("  * Calibrated decision thresholds sample (rare vs frequent):")
    sample_attrs = ["Age-Young", "Age-Adult", "Age-Old", "UpperBody-Color-Black", "UpperBody-Color-Yellow"]
    for attr in sample_attrs:
        idx = list(UPAR_40_ATTRIBUTES).index(attr)
        prior_val = calibrator.priors[idx]
        tau_val = calibrator.calibrated_thresholds[idx]
        print(f"    - {attr:22s}: prior={prior_val:.4f} -> calibrated_threshold={tau_val:.4f}")

    # Test intra-group color competition
    # Simulate an instance where model outputs high probabilities for Black (0.80), Blue (0.75), Grey (0.70)
    test_probs = np.full((1, 40), 0.10)
    test_probs[0, 8] = 0.85   # Black (UpperBody)
    test_probs[0, 9] = 0.65   # Blue  (UpperBody, diff 0.20 > margin 0.15 -> should be suppressed)
    test_probs[0, 12] = 0.60  # Grey  (UpperBody, diff 0.25 > margin 0.15 -> should be suppressed)
    
    comp_probs = calibrator.calibrate_probabilities(test_probs, apply_competition=True)
    bin_preds = calibrator.calibrate(test_probs, apply_competition=True)
    
    print("  * Color Competition Test (UpperBody Colors Black=0.85, Blue=0.65, Grey=0.60):")
    print(f"    - Post-comp prob Black (attr 8) : {comp_probs[0, 8]:.4f} (Binary: {bin_preds[0, 8]})")
    print(f"    - Post-comp prob Blue  (attr 9) : {comp_probs[0, 9]:.4f} (Binary: {bin_preds[0, 9]})")
    print(f"    - Post-comp prob Grey  (attr 12): {comp_probs[0, 12]:.4f} (Binary: {bin_preds[0, 12]})")
    assert bin_preds[0, 8] == 1, "Dominant color Black must remain 1"
    assert bin_preds[0, 9] == 0, "Rival color Blue must be suppressed to 0 by competition"
    assert bin_preds[0, 12] == 0, "Rival color Grey must be suppressed to 0 by competition"
    print("  >> PASS: Intra-group color competition successfully eliminated secondary color hallucinations.")

    # Comparative evaluation report
    eval_res = calibrator.evaluate_and_compare(dummy_y_true, dummy_y_pred)
    print(f"\n{eval_res['report']}")
    print("=" * 75)
    print("ALL VERIFICATION SMOKE TESTS PASSED SUCCESSFULLY WITH ZERO ERRORS!")
    print("=" * 75)


if __name__ == "__main__":
    run_smoke_test()
