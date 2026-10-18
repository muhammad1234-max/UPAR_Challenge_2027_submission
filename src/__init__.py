"""
src package initialization for UPAR Challenge 2027.
"""
from src.models_jev_asrhc import (
    JEVMultiSplitEngine,
    AnatomicalStripHead,
    BaselineGAPHead,
    CrossStripAttentionRefiner,
    create_jev_asrhc_model,
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
    export_codabench_submission,
    export_codabench_code_submission,
    run_ablation_study,
)

__all__ = [
    "JEVMultiSplitEngine",
    "AnatomicalStripHead",
    "BaselineGAPHead",
    "CrossStripAttentionRefiner",
    "create_jev_asrhc_model",
    "UPAR_40_ATTRIBUTES",
    "compute_upar_metrics",
    "SourcePriorHarmonicCalibrator",
    "UPARCsvDataset",
    "train_one_model",
    "run_jev_inference",
    "export_codabench_submission",
    "export_codabench_code_submission",
    "run_ablation_study",
]
