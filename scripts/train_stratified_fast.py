"""
train_stratified_fast.py
========================
High-Throughput Cloud GPU (Google Colab T4 / Kaggle) Fast Stratified Training
and Ablation Benchmarking Script for UPAR Challenge (WACV 2027) Track 1.

Features:
---------
1. Balanced Domain Stratification:
   Automatically samples balanced quotas across PA-100K (79k), Market1501 (10k),
   and PETA (8.6k) to prevent PA-100K from dominating gradient updates.
   Default: 15,000 samples (5,000 per domain), finishing 5 epochs in < 15 min on T4 GPU.
   Supports full training on all 97,669 images with `--stratified-train-samples 0`.
2. Cloud GPU Optimization:
   - PyTorch AMP mixed-precision acceleration (torch.cuda.amp)
   - Multi-worker threaded data loading with pinned memory
   - Shared-stem Joint Execution Vectorization (JEV) inference
3. End-to-End Evaluation & Submission Generation:
   - Computes official Track 1 Harmonic Mean (HM), label mA, and instance F1.
   - Generates 4-row ablation study table (CSV and Markdown).
   - Packages official CodaBench submission archives:
     * submission_track1_baseline.zip (Results CSV)
     * submission_track1_asrhc.zip (Results CSV)
     * codabench_code_submission_asrhc.zip (Complete code worker package)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

# Ensure workspace root and src directory are in sys.path
_FILE_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _FILE_DIR.parent
for _p in [str(_WORKSPACE_ROOT), str(_WORKSPACE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.models_jev_asrhc import (
    JEVMultiSplitEngine,
    UPAR_40_ATTRIBUTES,
)
from src.metrics_and_calibration import (
    compute_upar_metrics,
    SourcePriorHarmonicCalibrator,
)
from src.run_pipeline import (
    UPARCsvDataset,
    train_one_model,
    run_jev_inference,
    run_ablation_study,
    export_codabench_submission,
    export_codabench_code_submission,
)


def get_domain_stratified_indices(
    df: pd.DataFrame,
    total_samples: int = 15000,
    seed: int = 42,
) -> List[int]:
    """
    Extracts balanced stratified sample indices across PA-100K, Market1501, and PETA.
    """
    np.random.seed(seed)
    img_col = "# image" if "# image" in df.columns else df.columns[0]
    
    # Extract domain prefix
    domains = df[img_col].apply(lambda x: str(x).split("/")[0])
    unique_domains = domains.unique().tolist()
    
    counts = {d: (domains == d).sum() for d in unique_domains}
    print(f"  Source Domain Population: {counts}")
    
    # Target quota per domain
    num_domains = len(unique_domains)
    quota_per_domain = total_samples // num_domains
    
    selected_indices: List[int] = []
    allocated = 0
    
    # First pass: allocate up to quota
    for d in unique_domains:
        d_idx = df.index[domains == d].to_numpy()
        take = min(quota_per_domain, len(d_idx))
        chosen = np.random.choice(d_idx, size=take, replace=False)
        selected_indices.extend(chosen.tolist())
        allocated += take
        
    # Second pass: allocate remainder from largest datasets if any quota remained
    remainder = total_samples - allocated
    if remainder > 0:
        remaining_pool = list(set(df.index) - set(selected_indices))
        if len(remaining_pool) > 0:
            extra = np.random.choice(remaining_pool, size=min(remainder, len(remaining_pool)), replace=False)
            selected_indices.extend(extra.tolist())
            
    np.random.shuffle(selected_indices)
    return selected_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UPAR Challenge 2027: Fast Stratified Cloud GPU Training & Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-csv", type=str, default="data/annotations/task1/train/gt.csv",
                        help="Path to training annotations gt.csv")
    parser.add_argument("--val-csv", type=str, default="data/annotations/task1/val/gt.csv",
                        help="Path to validation annotations gt.csv")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root directory for dataset image crops")
    parser.add_argument("--stratified-train-samples", type=int, default=15000,
                        help="Number of balanced stratified training samples across PA-100K, Market, and PETA. Set to 0 for full 97,669 images.")
    parser.add_argument("--val-samples", type=int, default=0,
                        help="Number of validation samples to evaluate. Set to 0 for full 33,407 validation images.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training and inference (64 optimal for 16GB T4 GPU)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to train each head")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Base learning rate")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                        help="Load ImageNet pretrained ConvNeXt-Base backbone weights")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of DataLoader workers (defaults to 2 for Colab 2-CPU compatibility)")
    parser.add_argument("--train-head", type=str, default="both", choices=["both", "baseline", "asr"],
                        help="Which heads to train: 'both', 'baseline', or 'asr' (useful to resume ASR head if baseline is already trained)")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Path to an existing model checkpoint to resume from")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to save tables, predictions, and logs")
    parser.add_argument("--drive-backup-dir", type=str, default=None,
                        help="Optional Google Drive path (e.g. /content/drive/MyDrive/UPAR_Challenge_2027) to automatically mirror all results and checkpoints upon completion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    tables_dir = results_dir / "tables"
    preds_dir = results_dir / "predictions"
    ckpt_dir = results_dir / "checkpoints"
    tables_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        device_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  [GPU Acceleration Active] Detected {device_name} ({gpu_mem:.2f} GB VRAM)")
    else:
        print("  [Warning] CUDA not detected; running in CPU mode. Reduce epochs or samples.")
        
    print("\n" + "=" * 80)
    print("UPAR CHALLENGE 2027: ASR-HC STRATIFIED FAST CLOUD GPU TRAINING")
    print("=" * 80)
    
    # --------------------------------------------------------------------------
    # Step 1: Dataset Loading & Stratified Subsetting
    # --------------------------------------------------------------------------
    print(f"\n[1/6] Loading Full Training Dataset: {args.train_csv}")
    train_full_df = pd.read_csv(args.train_csv)
    total_available_train = len(train_full_df)
    
    # Check if stratified subsetting is requested
    if args.stratified_train_samples > 0 and args.stratified_train_samples < total_available_train:
        print(f"  Applying Balanced Domain Stratification ({args.stratified_train_samples} samples across domains):")
        stratified_idx = get_domain_stratified_indices(
            train_full_df, total_samples=args.stratified_train_samples
        )
        train_dataset = UPARCsvDataset(
            csv_path=args.train_csv,
            data_root=args.data_root,
            is_train=True,
        )
        train_dataset = Subset(train_dataset, stratified_idx)
        print(f"  -> Stratified training subset created with {len(train_dataset)} balanced samples.")
    else:
        train_dataset = UPARCsvDataset(
            csv_path=args.train_csv,
            data_root=args.data_root,
            is_train=True,
        )
        print(f"  -> Full training dataset loaded ({len(train_dataset)} samples).")
        
    # Validation dataset
    val_max = args.val_samples if args.val_samples > 0 else None
    print(f"\n[2/6] Loading Validation Dataset: {args.val_csv} (max: {val_max or 'ALL 33,407'})")
    val_dataset = UPARCsvDataset(
        csv_path=args.val_csv,
        data_root=args.data_root,
        is_train=False,
        max_samples=val_max,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"  Dataloaders ready: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")
    
    # --------------------------------------------------------------------------
    # Step 2: Strictly Training-Split Positive Priors (Zero Leakage)
    # --------------------------------------------------------------------------
    print("\n[3/6] Computing Strictly Training-Split Positive Priors pi_j...")
    calibrator = SourcePriorHarmonicCalibrator(alpha=0.35, color_margin=0.15)
    # Calculate priors from the complete training annotations for maximum statistical power
    attr_cols = [c for c in train_full_df.columns if c != "# image"]
    full_train_priors = train_full_df[attr_cols].mean().to_dict()
    calibrator.set_priors(full_train_priors)
    print(f"  Training priors calculated for {len(full_train_priors)} attributes with zero validation leakage.")
    
    # --------------------------------------------------------------------------
    # Step 3: Model Initialization & Multi-Head Training
    # --------------------------------------------------------------------------
    print(f"\n[4/6] Instantiating JEVMultiSplitEngine (Backbone: ConvNeXt-Base, Pretrained={args.pretrained})...")
    model = JEVMultiSplitEngine(pretrained=args.pretrained, num_classes=40)
    model.to(device)
    
    # Check for resume checkpoint or pre-trained baseline
    base_ckpt_path = ckpt_dir / "checkpoint_after_baseline.pth"
    if args.resume_checkpoint and Path(args.resume_checkpoint).is_file():
        print(f"  Loading resume checkpoint: {args.resume_checkpoint}")
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt, strict=False)
    elif args.train_head == "asr" and base_ckpt_path.is_file():
        print(f"  Loading pre-trained baseline checkpoint: {base_ckpt_path}")
        model.load_state_dict(torch.load(base_ckpt_path, map_location=device), strict=False)
        
    # 3a. Train Baseline Head
    if args.train_head in ("both", "baseline"):
        print("\n--- Phase A: Training Baseline GAP Head ---")
        train_one_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            head_type="baseline",
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            use_amp=(device.type == "cuda"),
        )
        torch.save(model.state_dict(), base_ckpt_path)
        print(f"  Saved baseline checkpoint to: {base_ckpt_path}")
    else:
        print("\n--- Phase A: Skipping Baseline GAP Training (using loaded weights) ---")
    
    # 3b. Train ASR-HC Head
    if args.train_head in ("both", "asr"):
        print("\n--- Phase B: Training Anatomical Strip-Routed Head (ASR-HC) ---")
        train_one_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            head_type="asr",
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            use_amp=(device.type == "cuda"),
        )
        torch.save(model.state_dict(), ckpt_dir / "checkpoint_after_asr.pth")
    else:
        print("\n--- Phase B: Skipping ASR Head Training ---")
    
    # --------------------------------------------------------------------------
    # Step 4: Joint Execution Vectorization (JEV) Inference
    # --------------------------------------------------------------------------
    print("\n[5/6] Executing Vectorized JEV Inference on Validation Split...")
    jev_out = run_jev_inference(model, val_loader, device=device)
    
    val_targets = jev_out["targets"]
    baseline_probs = jev_out["baseline_probs"]
    asr_probs = jev_out["asr_probs"]
    image_paths = jev_out["image_paths"]
    
    # --------------------------------------------------------------------------
    # Step 5: 4-Row Ablation Study Benchmark
    # --------------------------------------------------------------------------
    csv_table_path = tables_dir / "ablation_results.csv"
    md_table_path = tables_dir / "ablation_table.md"
    run_ablation_study(
        y_true=val_targets,
        baseline_probs=baseline_probs,
        asr_probs=asr_probs,
        calibrator=calibrator,
        output_csv_path=csv_table_path,
        output_md_path=md_table_path,
    )
    
    # --------------------------------------------------------------------------
    # Step 6: Package CodaBench Submissions
    # --------------------------------------------------------------------------
    print("\n[6/6] Generating CodaBench Official Submission Packages...")
    
    # 1. Baseline submission archive
    base_zip = preds_dir / "submission_track1_baseline.zip"
    export_codabench_submission(
        image_paths=image_paths,
        predictions=baseline_probs,
        output_zip_path=base_zip,
    )
    
    # 2. ASR-HC calibrated predictions archive (continuous warping for CodaBench > 0.5)
    calib_asr_preds = calibrator.calibrate_continuous_for_05(asr_probs, apply_competition=True)
    asrhc_zip = preds_dir / "submission_track1_asrhc.zip"
    export_codabench_submission(
        image_paths=image_paths,
        predictions=calib_asr_preds,
        output_zip_path=asrhc_zip,
    )
    
    # 3. Complete CodaBench code submission package
    code_sub_zip = preds_dir / "codabench_code_submission_asrhc.zip"
    export_codabench_code_submission(
        model=model,
        calibrator=calibrator,
        output_zip_path=code_sub_zip,
    )
    
    # 4. Save Standalone Trained Model Checkpoint
    model_ckpt_path = ckpt_dir / "model_asrhc_fp16.pth"
    fp16_state = {k: (v.half().cpu() if torch.is_floating_point(v) else v.cpu()) for k, v in model.state_dict().items()}
    torch.save(fp16_state, model_ckpt_path)
    print(f"  * Model Checkpoint : {model_ckpt_path}")
    
    # 5. Optional Auto-Backup to Google Drive
    if args.drive_backup_dir:
        drive_dest = Path(args.drive_backup_dir)
        try:
            drive_dest.mkdir(parents=True, exist_ok=True)
            import shutil
            target_results = drive_dest / "results"
            shutil.copytree(results_dir, target_results, dirs_exist_ok=True)
            print(f"\n[GOOGLE DRIVE BACKUP SUCCESS] All tables, checkpoints, and submissions saved to:\n  {target_results.resolve()}")
        except Exception as e:
            print(f"\n[GOOGLE DRIVE BACKUP NOTICE] Failed to auto-backup to {args.drive_backup_dir}: {e}")
            
    print("\n" + "=" * 80)
    print("STRATIFIED FAST TRAINING PIPELINE COMPLETE!")
    print(f"  * Ablation Table   : {csv_table_path}")
    print(f"  * Results Archive  : {asrhc_zip}")
    print(f"  * Code Worker Pkg  : {code_sub_zip}")
    print(f"  * Standalone Checkpoint: {model_ckpt_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
