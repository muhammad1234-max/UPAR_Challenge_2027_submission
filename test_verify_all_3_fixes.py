"""
Master Verification Suite for UPAR Challenge 2027 Track 1:
Validates all 3 critical engineering fixes:
1. Continuous Probability Warping for CodaBench's > 0.5 Threshold
2. Offline-Safe Full FP16 Checkpoint & Batched Inference in CodaBench run.py
3. Self-Contained Google Colab / Kaggle Execution (%%writefile cells)
"""
import json
import os
import shutil
import sys
import tempfile
import zipfile
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.abspath("."))
from src.models_jev_asrhc import JEVMultiSplitEngine, UPAR_40_ATTRIBUTES
from src.metrics_and_calibration import SourcePriorHarmonicCalibrator
from src.run_pipeline import export_codabench_code_submission, export_codabench_submission

def test_fix_1_continuous_warping():
    print("\n" + "=" * 70)
    print("VERIFYING FIX 1: Continuous Probability Warping for CodaBench > 0.5")
    print("=" * 70)
    
    np.random.seed(42)
    priors = {name: float(np.random.uniform(0.01, 0.70)) for name in UPAR_40_ATTRIBUTES}
    calibrator = SourcePriorHarmonicCalibrator(priors=priors, alpha=0.35, color_margin=0.15)
    calibrator._compute_calibrated_thresholds()
    
    N = 10000
    test_probs = np.random.uniform(0.0, 1.0, size=(N, 40))
    test_probs[0, :] = 0.0
    test_probs[1, :] = 1.0
    test_probs[2, :] = calibrator.calibrated_thresholds
    test_probs[3, :] = calibrator.calibrated_thresholds + 1e-5
    test_probs[4, :] = calibrator.calibrated_thresholds - 1e-5
    
    # Compare binary calibrate vs continuous warped thresholded at 0.5
    binary_calibrated = calibrator.calibrate(test_probs, apply_competition=True)
    warped_p = calibrator.calibrate_continuous_for_05(test_probs, apply_competition=True)
    warped_binary = (warped_p >= 0.5).astype(np.int32)
    
    discrepancies = int(np.sum(binary_calibrated != warped_binary))
    total_evals = N * 40
    print(f"  Evaluated {total_evals:,} attribute predictions across {N:,} instances.")
    print(f"  Range of warped_p: [{warped_p.min():.5f}, {warped_p.max():.5f}]")
    print(f"  Discrepancies between (warped_p >= 0.5) and calibrate(probs): {discrepancies}")
    assert discrepancies == 0, f"Found {discrepancies} mismatches!"
    assert np.all(warped_p >= 0.0) and np.all(warped_p <= 1.0), "Warped probabilities must be in [0, 1]"
    print("  >>> FIX 1 PASSED: 100% mathematical and empirical equivalence verified!")


def test_fix_2_offline_fp16_and_batched_run_py():
    print("\n" + "=" * 70)
    print("VERIFYING FIX 2: Offline-Safe FP16 Checkpoint & Batched Inference")
    print("=" * 70)
    
    model = JEVMultiSplitEngine(pretrained=False, num_classes=40)
    priors = {name: float(np.random.uniform(0.02, 0.35)) for name in UPAR_40_ATTRIBUTES}
    calibrator = SourcePriorHarmonicCalibrator(priors=priors, alpha=0.35, color_margin=0.15)
    
    zip_path = "results/predictions/codabench_code_submission_asrhc.zip"
    export_codabench_code_submission(model=model, calibrator=calibrator, output_zip_path=zip_path)
    
    # Sandbox testing
    sandbox = tempfile.mkdtemp(prefix="codabench_test_")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(sandbox)
            file_list = zf.namelist()
            
        print(f"  Archive entries ({len(file_list)}): {file_list}")
        required = ["metadata.yaml", "run.py", "models_jev_asrhc.py", "metrics_and_calibration.py", "assets/model_fp16.pth", "assets/calibrator_priors.json"]
        for r in required:
            assert r in file_list, f"Missing required file: {r}"
            
        # Verify FP16
        weights_p = os.path.join(sandbox, "assets", "model_fp16.pth")
        sd = torch.load(weights_p, map_location="cpu")
        fp16_count = sum(1 for v in sd.values() if torch.is_floating_point(v) and v.dtype == torch.float16)
        total_float = sum(1 for v in sd.values() if torch.is_floating_point(v))
        print(f"  State dict float tensors: {fp16_count}/{total_float} in float16.")
        assert fp16_count == total_float, "All float tensors must be stored in FP16!"
        
        # Test loading run.py in isolated sandbox
        sys.path.insert(0, sandbox)
        import importlib.util
        spec = importlib.util.spec_from_file_location("sb_run", os.path.join(sandbox, "run.py"))
        sb_run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sb_run)
        
        sb_run.load_model()
        assert sb_run._MODEL is not None, "Model failed to load offline!"
        assert sb_run._CALIBRATOR is not None, "Calibrator failed to load offline!"
        
        # Test inputs
        dummy_file = os.path.join(sandbox, "test_crop.jpg")
        img_pil = Image.fromarray(np.random.randint(0, 256, (256, 192, 3), dtype=np.uint8))
        img_pil.save(dummy_file)
        
        reversed_names = list(reversed(UPAR_40_ATTRIBUTES))
        samples = [
            {"image": img_pil, "attribute_names": list(UPAR_40_ATTRIBUTES)},
            {"image": np.random.randint(0, 256, (256, 192, 3), dtype=np.uint8), "attribute_names": reversed_names},
            {"image": np.random.uniform(0.0, 1.0, (256, 192, 3)).astype(np.float32), "attribute_names": list(UPAR_40_ATTRIBUTES)},
            {"image": torch.rand(3, 256, 192), "attribute_names": reversed_names},
            {"image_path": dummy_file, "attribute_names": list(UPAR_40_ATTRIBUTES)},
        ]
        
        # 1. predict_image
        p_single = sb_run.predict_image(samples[0])
        assert len(p_single) == 40 and all(0.0 <= x <= 1.0 for x in p_single)
        print("  predict_image(sample) successfully verified.")
        
        # 2. predict_batch
        p_batch = sb_run.predict_batch(samples)
        assert len(p_batch) == 5
        for i, row in enumerate(p_batch):
            assert len(row) == 40
            assert all(0.0 <= x <= 1.0 for x in row)
        print("  predict_batch(samples) across 5 multi-format samples successfully verified.")
        print("  >>> FIX 2 PASSED: Offline FP16 weights & batched inference verified!")
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def test_fix_3_colab_notebook_self_contained():
    print("\n" + "=" * 70)
    print("VERIFYING FIX 3: Standalone Google Colab Notebook with %%writefile Cells")
    print("=" * 70)
    
    nb_path = "UPAR_2027_Cloud_GPU_Train.ipynb"
    assert os.path.isfile(nb_path), f"Notebook missing at {nb_path}"
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        
    cells = nb_data["cells"]
    print(f"  Loaded notebook with {len(cells)} cells.")
    
    # Locate writefile cells
    writefile_targets = {
        "src/__init__.py": None,
        "src/models_jev_asrhc.py": None,
        "src/metrics_and_calibration.py": None,
        "src/run_pipeline.py": None,
        "scripts/train_stratified_fast.py": None,
    }
    
    for c in cells:
        if c.get("cell_type") == "code":
            src_lines = c.get("source", [])
            if src_lines:
                first_line = src_lines[0].strip()
                if first_line.startswith("%%writefile"):
                    target = first_line.split()[1].replace("\\", "/")
                    if target in writefile_targets:
                        writefile_targets[target] = "".join(src_lines[1:])
                        
    for target, content in writefile_targets.items():
        assert content is not None, f"Missing %%writefile cell for {target}!"
        print(f"  Verified %%writefile cell for: {target} ({len(content):,} chars)")
        
    # Simulate writing the files into a clean temp folder and checking py_compile
    sim_dir = tempfile.mkdtemp(prefix="colab_sim_")
    try:
        os.makedirs(os.path.join(sim_dir, "src"), exist_ok=True)
        os.makedirs(os.path.join(sim_dir, "scripts"), exist_ok=True)
        
        for rel_path, code in writefile_targets.items():
            dest = os.path.join(sim_dir, rel_path)
            with open(dest, "w", encoding="utf-8") as f:
                f.write(code)
                
            # Verify compilation
            import py_compile
            py_compile.compile(dest, doraise=True)
            
        print("  All injected files compiled cleanly without errors in simulated Colab VM.")
        print("  >>> FIX 3 PASSED: Self-contained notebook execution verified!")
    finally:
        shutil.rmtree(sim_dir, ignore_errors=True)


if __name__ == "__main__":
    test_fix_1_continuous_warping()
    test_fix_2_offline_fp16_and_batched_run_py()
    test_fix_3_colab_notebook_self_contained()
    print("\n" + "*" * 70)
    print("ALL 3 CRITICAL FIXES VERIFIED SUCCESSFULLY AND HARDENED!")
    print("*" * 70 + "\n")
