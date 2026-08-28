"""Full-grid MCMC inference for BGM-IV.

Submodules load lazily because importing the sampler changes TensorFlow
determinism settings and must happen only when MCMC starts.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SUBMODULES = ("target", "sampler", "readout", "inference")

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
    "resolve_target": "target",
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
    "FullGridReadout": "readout",
    "ReadoutConfig": "readout",
    "ReadoutError": "readout",
    "StructuralQueryTable": "readout",
    "build_query_table": "readout",
    "gaussian_mixture_quantiles": "readout",
    "FAMILY_RECIPES": "inference",
    "FamilyRecipe": "inference",
    "MCMCConfig": "inference",
    "MCMCInferenceError": "inference",
    "derive_mcmc_seeds": "inference",
    "estimate_pilot_state_variance": "inference",
    "execution_environment": "inference",
    "run_mcmc": "inference",
    "run_mcmc_grid": "inference",
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
