"""Certified posterior integration for BGM-IV.

Submodules load lazily: ``sampler`` changes global TensorFlow execution
settings at import (TensorFloat-32 off, op determinism on), and that must
only happen once sampling starts.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SUBMODULES = ("target", "sampler", "diagnostics", "readout", "certify")

_EXPORTS = {
    "AffinePreprocessorSpec": "target",
    "EvidenceBlockSpec": "target",
    "FeaturePreprocessorSpec": "target",
    "ModelTrainingProvenance": "target",
    "PCAFeaturePreprocessorSpec": "target",
    "ResolvedTarget": "target",
    "TargetSpec": "target",
    "independent_formula_oracle": "target",
    "model_training_provenance": "target",
    "require_digest": "target",
    "resolve_target": "target",
    "sha256_array": "target",
    "sha256_decoder": "target",
    "sha256_json": "target",
    "sha256_weights": "target",
    "FrozenBatch": "sampler",
    "FrozenEpoch": "sampler",
    "FrozenHMCError": "sampler",
    "FrozenHMCNumericalError": "sampler",
    "FrozenVectorizedHMC": "sampler",
    "GaussianContextEvaluator": "sampler",
    "LatentPosteriorEvaluator": "sampler",
    "MassRegularization": "sampler",
    "ProductionConfig": "sampler",
    "ProductionSegment": "sampler",
    "TrajectoryPolicy": "sampler",
    "WarmupConfig": "sampler",
    "WarmupResult": "sampler",
    "assert_axis_separable": "sampler",
    "overdispersed_initial_state": "sampler",
    "regularize_state_variance": "sampler",
    "Action": "diagnostics",
    "Assessment": "diagnostics",
    "BatchIdentity": "diagnostics",
    "GateError": "diagnostics",
    "OutcomeTransform": "diagnostics",
    "PrecisionPolicy": "diagnostics",
    "SamplerEvidence": "diagnostics",
    "StructuralMetric": "diagnostics",
    "assess_sampler": "diagnostics",
    "build_batch_record": "diagnostics",
    "chain_diagnostics": "diagnostics",
    "choose_block_len": "diagnostics",
    "functional_iact": "diagnostics",
    "score_batch": "diagnostics",
    "structural_metric": "diagnostics",
    "structural_metric_evidence": "diagnostics",
    "SCORING_UNITS": "readout",
    "CalibrationConfig": "readout",
    "FunctionalReadout": "readout",
    "PredictiveCalibrationAccumulator": "readout",
    "ReadoutError": "readout",
    "StructuralQueryTable": "readout",
    "batch_query_view": "readout",
    "build_query_table": "readout",
    "gaussian_mixture_quantiles": "readout",
    "FAMILY_RECIPES": "certify",
    "GATE_STATES": "certify",
    "BatchOutcome": "certify",
    "CertificationError": "certify",
    "FamilyRecipe": "certify",
    "MCMCConfig": "certify",
    "aggregate_batch_outcomes": "certify",
    "certify_grid": "certify",
    "derive_certification_seeds": "certify",
    "estimate_pilot_state_variance": "certify",
    "execution_environment": "certify",
    "run_mcmc": "certify",
}

__all__ = list(_SUBMODULES) + sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return import_module(f"{__name__}.{name}")
    owner = _EXPORTS.get(name)
    if owner is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{owner}"), name)


def __dir__():
    return sorted(__all__)
