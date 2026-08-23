"""Latent posterior targets of fitted BGM-IV models.

A fitted model and a covariate preprocessor resolve once into an immutable
:class:`TargetSpec`; its identity hashes the decoder state and source, the
preprocessor and every evidence block (all event sums with unit powers), so a
posterior draw is always attributable to one exact target.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from typing import Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf

from ..hashing import require_digest, sha256_array, sha256_json, sha256_weights


# --- covariate preprocessors -------------------------------------------------

@dataclass(frozen=True)
class AffinePreprocessorSpec:
    """The exact raw-v -> model-v affine transform used at inference."""

    mean: np.ndarray
    scale: np.ndarray
    name: str = "affine"

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=np.float32).reshape(-1)
        scale = np.asarray(self.scale, dtype=np.float32).reshape(-1)
        if mean.shape != scale.shape or not len(mean):
            raise ValueError("preprocessor mean/scale must be non-empty equal vectors")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("preprocessor parameters must be finite")
        if np.any(scale <= 0):
            raise ValueError("preprocessor scale must be strictly positive")

    @property
    def dimension(self) -> int:
        return int(np.asarray(self.mean).size)

    @property
    def identity(self) -> str:
        return sha256_json("preprocessor",
            {
                "kind": "affine",
                "name": self.name,
                "dimension": self.dimension,
                "mean_hash": sha256_array(
                    np.asarray(self.mean, dtype=np.float32).reshape(-1)
                ),
                "scale_hash": sha256_array(
                    np.asarray(self.scale, dtype=np.float32).reshape(-1)
                ),
            }
        )

    def transform(self, raw_v):
        """Apply the declared transform and return model-scale float32 data."""
        raw = np.asarray(raw_v, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != self.dimension:
            raise ValueError("raw_v shape does not match preprocessor dimension")
        mean = np.asarray(self.mean, dtype=np.float32).reshape(1, -1)
        scale = np.asarray(self.scale, dtype=np.float32).reshape(1, -1)
        return ((raw - mean) / scale).astype(np.float32)

    @property
    def model_dimension(self) -> int:
        """Width of the MODEL-scale covariate the transform produces."""
        return self.dimension

    @classmethod
    def identity_map(cls, dimension: int, name: str = "identity"):
        return cls(
            mean=np.zeros(int(dimension), dtype=np.float32),
            scale=np.ones(int(dimension), dtype=np.float32),
            name=name,
        )


class FeaturePreprocessorSpec:
    """Map raw image covariates into the model representation.

    The identity binds the encoder, preprocessing statistics, and source
    checkpoint used by the transform.
    """

    kind = "image-representation"

    def __init__(
        self,
        *,
        trunk,
        mean,
        scale,
        input_dimension: int,
        feature_slice: slice = slice(1, 65),
        noise_slice: Optional[slice] = None,
        trunk_architecture: str = "DemandImageFeatureExtractor",
        pixel_checkpoint: Optional[Mapping[str, str]] = None,
        name: str = "image_representation",
        provenance: Optional[Mapping[str, str]] = None,
    ):
        self.trunk = trunk
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.scale = np.asarray(scale, dtype=np.float32).reshape(-1)
        if self.mean.shape != self.scale.shape or not len(self.mean):
            raise ValueError("feature mean/scale must be non-empty equal vectors")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)):
            raise ValueError("feature standardizer must be finite")
        if np.any(self.scale <= 0):
            raise ValueError("feature scale must be strictly positive")
        self.input_dimension = int(input_dimension)
        self.feature_slice = feature_slice
        self.noise_slice = noise_slice
        self.trunk_architecture = str(trunk_architecture)
        self.pixel_checkpoint = (
            None if pixel_checkpoint is None else {str(k): str(v) for k, v in pixel_checkpoint.items()}
        )
        self.name = str(name)
        self.provenance = {} if provenance is None else {str(k): str(v) for k, v in provenance.items()}
        self.feature_dim = int(self.mean.size)
        if (feature_slice.stop - feature_slice.start) != self.feature_dim:
            raise ValueError("feature_slice width must equal the standardizer width")
        self.noise_dim = 0
        if noise_slice is not None:
            if noise_slice.stop is not None:
                self.noise_dim = int(noise_slice.stop - noise_slice.start)
            else:
                self.noise_dim = int(self.input_dimension - noise_slice.start)
            if self.noise_dim <= 0:
                raise ValueError("noise_slice must select at least one dimension")
        self.trunk_weights_sha256 = sha256_weights(trunk)
        self.trunk_num_params = int(
            sum(int(np.prod(w.shape)) for w in getattr(trunk, "weights", ()))
        )

    @property
    def dimension(self) -> int:
        """Width of the RAW input rows this transform accepts."""
        return self.input_dimension

    @property
    def model_dimension(self) -> int:
        return 1 + self.feature_dim + self.noise_dim

    @property
    def identity(self) -> str:
        return sha256_json("preprocessor",
            {
                "kind": self.kind,
                "name": self.name,
                "input_dimension": self.input_dimension,
                "model_dimension": self.model_dimension,
                "feature_slice": [int(self.feature_slice.start), int(self.feature_slice.stop)],
                "noise_dim": int(self.noise_dim),
                "trunk_architecture": self.trunk_architecture,
                "trunk_weights_sha256": self.trunk_weights_sha256,
                "trunk_num_params": self.trunk_num_params,
                "inference_mode": True,
                "unique_rows_single_pass": True,
                "mean_hash": sha256_array(self.mean),
                "scale_hash": sha256_array(self.scale),
                "pixel_checkpoint": self.pixel_checkpoint,
                "provenance": self.provenance,
            }
        )

    def assert_runtime_identity(self) -> None:
        if sha256_weights(self.trunk) != self.trunk_weights_sha256:
            raise RuntimeError("image encoder weights changed after binding")

    def features(self, raw_v) -> np.ndarray:
        """Raw rows -> standardized phi ``[n, feature_dim]``.

        One trunk pass over the byte-unique raw rows, scattered back, so
        identical images map to bit-identical features (same contract as
        ``bgm_iv.features.image_encoder_features``).
        """
        raw = np.asarray(raw_v, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != self.input_dimension:
            raise ValueError("raw_v shape does not match preprocessor input dimension")
        self.assert_runtime_identity()
        order: dict = {}
        inverse = np.empty(raw.shape[0], np.int64)
        rows = []
        for index in range(raw.shape[0]):
            key = raw[index].tobytes()
            position = order.get(key)
            if position is None:
                position = len(rows)
                order[key] = position
                rows.append(raw[index])
            inverse[index] = position
        output = self.trunk(tf.constant(np.stack(rows, axis=0)), training=False)
        phi = np.asarray(output, np.float32)[:, self.feature_slice]
        if phi.shape[1] != self.feature_dim:
            raise ValueError("trunk output width does not match the feature slice")
        phi = phi[inverse]
        return ((phi - self.mean[None, :]) / self.scale[None, :]).astype(np.float32)

    def transform(self, raw_v):
        raw = np.asarray(raw_v, dtype=np.float32)
        parts = [raw[:, :1], self.features(raw)]
        if self.noise_slice is not None:
            parts.append(raw[:, self.noise_slice])
        model_v = np.concatenate(parts, axis=1).astype(np.float32)
        if model_v.shape[1] != self.model_dimension:
            raise ValueError("transformed covariate width mismatch")
        return model_v

    def to_payload(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "identity": self.identity,
            "input_dimension": self.input_dimension,
            "model_dimension": self.model_dimension,
            "feature_dim": self.feature_dim,
            "noise_dim": self.noise_dim,
            "trunk_architecture": self.trunk_architecture,
            "trunk_weights_sha256": self.trunk_weights_sha256,
            "trunk_num_params": self.trunk_num_params,
            "mean_hash": sha256_array(self.mean),
            "scale_hash": sha256_array(self.scale),
            "pixel_checkpoint": self.pixel_checkpoint,
            "provenance": dict(self.provenance),
        }


class PCAFeaturePreprocessorSpec:
    """Map raw image covariates through the PCA control representation."""

    kind = "pca-feature"

    def __init__(
        self,
        *,
        components,
        pca_mean,
        mean,
        scale,
        input_dimension: int,
        pixel_slice: slice = slice(1, 785),
        noise_slice: Optional[slice] = None,
        name: str = "pca_feature",
    ):
        self.components = np.asarray(components, np.float32)
        self.pca_mean = np.asarray(pca_mean, np.float32).reshape(-1)
        self.mean = np.asarray(mean, np.float32).reshape(-1)
        self.scale = np.asarray(scale, np.float32).reshape(-1)
        if self.components.ndim != 2 or self.components.shape[1] != self.pca_mean.size:
            raise ValueError("components must be [k, pixels] matching pca_mean")
        if self.mean.shape != self.scale.shape or self.mean.size != self.components.shape[0]:
            raise ValueError("standardizer must be [k] matching the components")
        if np.any(self.scale <= 0) or not np.all(np.isfinite(self.scale)):
            raise ValueError("feature scale must be finite and positive")
        self.input_dimension = int(input_dimension)
        self.pixel_slice = pixel_slice
        self.noise_slice = noise_slice
        self.name = str(name)
        self.feature_dim = int(self.components.shape[0])
        self.noise_dim = 0
        if noise_slice is not None:
            stop = self.input_dimension if noise_slice.stop is None else noise_slice.stop
            self.noise_dim = int(stop - noise_slice.start)

    @property
    def dimension(self) -> int:
        return self.input_dimension

    @property
    def model_dimension(self) -> int:
        return 1 + self.feature_dim + self.noise_dim

    @property
    def identity(self) -> str:
        return sha256_json("preprocessor",
            {
                "kind": self.kind,
                "name": self.name,
                "input_dimension": self.input_dimension,
                "model_dimension": self.model_dimension,
                "components_hash": sha256_array(self.components),
                "pca_mean_hash": sha256_array(self.pca_mean),
                "mean_hash": sha256_array(self.mean),
                "scale_hash": sha256_array(self.scale),
                "pixel_scaling": "divide_by_255",
                "noise_dim": int(self.noise_dim),
            }
        )

    def features(self, raw_v) -> np.ndarray:
        raw = np.asarray(raw_v, np.float32)
        if raw.ndim != 2 or raw.shape[1] != self.input_dimension:
            raise ValueError("raw_v shape does not match preprocessor input dimension")
        images = raw[:, self.pixel_slice] / 255.0
        scores = (images - self.pca_mean[None, :]) @ self.components.T
        return ((scores - self.mean[None, :]) / self.scale[None, :]).astype(np.float32)

    def transform(self, raw_v):
        raw = np.asarray(raw_v, np.float32)
        parts = [raw[:, :1], self.features(raw)]
        if self.noise_slice is not None:
            parts.append(raw[:, self.noise_slice])
        return np.concatenate(parts, axis=1).astype(np.float32)

    def to_payload(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "identity": self.identity,
            "input_dimension": self.input_dimension,
            "model_dimension": self.model_dimension,
            "feature_dim": self.feature_dim,
            "noise_dim": self.noise_dim,
            "components_hash": sha256_array(self.components),
            "mean_hash": sha256_array(self.mean),
            "scale_hash": sha256_array(self.scale),
        }


@dataclass(frozen=True)
class ModelTrainingProvenance:
    """Training-data declaration of a checkpoint, bound to its decoder hash."""

    decoder_model_hash: str
    data_mode: str

    def __post_init__(self):
        if not self.decoder_model_hash or not self.data_mode:
            raise ValueError("model training provenance is incomplete")

    @property
    def identity(self) -> str:
        return sha256_json(
            "training-provenance",
            {"decoder_model_hash": self.decoder_model_hash, "data_mode": self.data_mode},
        )


@dataclass(frozen=True)
class EvidenceBlockSpec:
    name: str
    role: str
    event_size: int
    event_reduction: str
    power: float
    evidence_kind: str
    observation_support: str
    constants: str = "z-independent constants omitted"

    def __post_init__(self):
        if not self.name or self.role not in {"prior", "observation"}:
            raise ValueError("invalid evidence block name or role")
        if int(self.event_size) < 1:
            raise ValueError("event_size must be positive")
        if self.event_reduction != "sum":
            raise ValueError("every base evidence block must use event sum")
        if not np.isfinite(self.power) or self.power <= 0:
            raise ValueError("block power must be finite and positive")
        if self.evidence_kind not in {
            "log_prior",
            "log_likelihood",
            "generalized_score",
        }:
            raise ValueError("invalid evidence_kind")

    def payload(self):
        return {
            "name": self.name,
            "role": self.role,
            "event_size": int(self.event_size),
            "event_reduction": self.event_reduction,
            "power": float(self.power),
            "evidence_kind": self.evidence_kind,
            "observation_support": self.observation_support,
            "constants": self.constants,
        }


@dataclass(frozen=True)
class TargetSpec:
    """Immutable description of one latent posterior target.

    The identity hashes the family, every evidence block, the global power,
    the decoder state and source, the preprocessor and the training
    provenance; a resolved target is valid only while the model matches it.
    """

    family: str
    blocks: Tuple[EvidenceBlockSpec, ...]
    global_power: float
    decoder_model_hash: str
    preprocessor_hash: str
    model_training_provenance_hash: str
    use_bnn: bool
    dtype: str

    def __post_init__(self):
        if self.family not in {"demand", "vector"}:
            raise ValueError("unknown BGM-IV target family")
        if not self.blocks or len({block.name for block in self.blocks}) != len(self.blocks):
            raise ValueError("evidence block names must be non-empty and unique")
        if not np.isfinite(self.global_power) or self.global_power <= 0:
            raise ValueError("global_power must be finite and positive")
        if (
            not self.decoder_model_hash
            or not self.preprocessor_hash
            or not self.model_training_provenance_hash
        ):
            raise ValueError("model/preprocessor/training-provenance hashes are required")
        # A stochastic decoder has no fixed posterior unless a weight draw is frozen.
        if self.use_bnn:
            raise ValueError("Flipout decoder target is stochastic; freeze a draw first")
        if self.global_power != 1.0:
            raise ValueError("model posterior requires global_power=1")
        bad_power = [block.name for block in self.blocks if block.power != 1.0]
        if bad_power:
            raise ValueError(f"model posterior requires unit block powers; got {bad_power}")
        bad_kind = [
            block.name
            for block in self.blocks
            if block.evidence_kind not in {"log_prior", "log_likelihood"}
        ]
        if bad_kind:
            raise ValueError(f"model posterior cannot contain generalized scores: {bad_kind}")

    @property
    def identity(self) -> str:
        return sha256_json(
            "target",
            {
                "family": self.family,
                "blocks": [block.payload() for block in self.blocks],
                "global_power": float(self.global_power),
                "decoder_model_hash": self.decoder_model_hash,
                "preprocessor_hash": self.preprocessor_hash,
                "model_training_provenance_hash": self.model_training_provenance_hash,
                "use_bnn": bool(self.use_bnn),
                "dtype": self.dtype,
            },
        )

    @property
    def block_powers(self):
        return {block.name: float(block.power) for block in self.blocks}

    @property
    def manifest(self):
        """Identities that travel with every result produced from this target."""
        return {
            "target_hash": self.identity,
            "family": self.family,
            "global_power": float(self.global_power),
            "blocks": [block.payload() for block in self.blocks],
            "decoder_model_hash": self.decoder_model_hash,
            "preprocessor_hash": self.preprocessor_hash,
            "model_training_provenance_hash": self.model_training_provenance_hash,
        }


def _model_family(model) -> str:
    name = type(model).__name__
    if name == "BGM_IV_Vector":
        return "vector"
    if name == "BGM_IV_Image":
        return "mnist"
    if name == "BGM_IV":
        return "demand"
    raise TypeError(f"unsupported current model class: {name}")


def _float_or_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return float(value)


def _batchnorm_fused_flags(network) -> list:
    """Ordered ``fused`` flags of every BatchNormalization layer in a network."""
    flags = []
    for layer in getattr(network, "submodules", ()):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            fused = getattr(layer, "fused", None)
            flags.append(None if fused is None else bool(fused))
    return flags


def sha256_decoder(model) -> str:
    """Digest of the decoder architecture, its source primitives and every g_net variable."""
    family = _model_family(model)
    trainable_ids = {id(variable) for variable in model.g_net.trainable_variables}
    state = []
    for index, variable in enumerate(model.g_net.variables):
        value = np.asarray(variable.numpy())
        state.append(
            {
                "index": index,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "trainable": id(variable) in trainable_ids,
                "value_hash": sha256_array(value),
            }
        )
    try:
        source_objects = [
            type(model.g_net),
            type(model).get_log_covariate_posterior,
            model._continuous_sigma,
        ]
        if hasattr(type(model), "_covariate_loss_terms"):
            source_objects.append(type(model)._covariate_loss_terms)
        else:
            source_objects.append(model._gaussian_nll)
        primitive_source = "\n".join(
            inspect.getsource(source_object) for source_object in source_objects
        )
    except (OSError, TypeError) as exc:
        raise RuntimeError("cannot resolve decoder primitive source identity") from exc
    metadata = {
        "schema": "bgm-iv-decoder-state",
        "family": family,
        "decoder_class": type(model.g_net).__name__,
        "v_dim": int(model.params["v_dim"]),
        "z_dims": [int(value) for value in model.params["z_dims"]],
        "use_bnn": bool(model.params.get("use_bnn", False)),
        "vector_dim": int(getattr(model, "vector_dim", 0)),
        "image_dim": int(getattr(model, "image_dim", 0)),
        "extra_noise_dim": int(getattr(model, "extra_noise_dim", 0)),
        "sigma_v_override": (
            None
            if "sigma_v" not in model.params
            else float(model.params["sigma_v"])
        ),
        # Fixed time-noise override (vector/image analog of sigma_v): part of
        # the effective likelihood, hence identity-relevant.
        "sigma_time_override": (
            None
            if "sigma_time" not in model.params
            else float(model.params["sigma_time"])
        ),
        # Soft variance floors (learnable head bounded below): identity-
        # relevant exactly like the fixed overrides.
        "sigma_time_softfloor": (
            None
            if "sigma_time_softfloor" not in model.params
            else float(model.params["sigma_time_softfloor"])
        ),
        "sigma_v_softfloor": (
            None
            if "sigma_v_softfloor" not in model.params
            else float(model.params["sigma_v_softfloor"])
        ),
        # Vector-block floors (representation-space model) bound the decoded
        # ``vector_var``, so the floor (scalar or per-block) and the block
        # layout are part of the effective likelihood.
        "sigma_vector_softfloor": _float_or_list(
            model.params.get("sigma_vector_softfloor")
        ),
        "sigma_vector_override": (
            None
            if "sigma_vector" not in model.params
            else float(model.params["sigma_vector"])
        ),
        "vector_blocks": (
            None
            if model.params.get("vector_blocks") is None
            else [int(value) for value in model.params["vector_blocks"]]
        ),
        # BatchNormalization execution path of the decoder (image family):
        # fused and non-fused kernels are not bit-identical, and the flag is
        # not persisted in checkpoints, so it must be part of the identity.
        "batchnorm_fused_flags": _batchnorm_fused_flags(model.g_net),
        "primitive_source_hash": hashlib.sha256(
            primitive_source.encode()
        ).hexdigest(),
        "state": state,
    }
    return sha256_json("decoder", metadata)


def model_training_provenance(model) -> ModelTrainingProvenance:
    """Training-data declaration of a checkpoint, bound to its decoder hash."""
    family = _model_family(model)
    return ModelTrainingProvenance(
        decoder_model_hash=sha256_decoder(model),
        data_mode="soft_grayscale" if family == "mnist" else "not_applicable",
    )


def resolve_target(
    model,
    preprocessor: AffinePreprocessorSpec,
    *,
    global_power: float = 1.0,
    training_provenance: Optional[ModelTrainingProvenance] = None,
):
    """Resolve a fitted demand or vector model into an immutable target.

    Parameters
    ----------
    model : BGM_IV or BGM_IV_Vector
        Fitted model; its decoder is hashed into the identity.
    preprocessor : AffinePreprocessorSpec-like
        Raw-to-model covariate map whose ``identity`` is bound to the target.
    global_power : float
        Must be one for the model posterior.
    training_provenance : ModelTrainingProvenance, optional
        Defaults to the provenance derived from the model.

    Returns
    -------
    ResolvedTarget
    """
    family = _model_family(model)
    if family not in {"demand", "vector"}:
        raise ValueError(
            "only the demand and vector covariate posteriors are certifiable; "
            "the pixel-likelihood model has no model posterior target"
        )
    v_dim = int(model.params["v_dim"])
    z_dim = sum(int(value) for value in model.params["z_dims"])
    model_dimension = int(getattr(preprocessor, "model_dimension", preprocessor.dimension))
    if model_dimension != v_dim:
        raise ValueError("preprocessor dimension does not match model v_dim")
    if family == "vector" and str(model.params.get("covariate_block_scale", "sum")) != "sum":
        raise ValueError("the vector model must use event-sum covariate blocks")
    decoder_hash = sha256_decoder(model)
    if training_provenance is None:
        training_provenance = model_training_provenance(model)
    if training_provenance.decoder_model_hash != decoder_hash:
        raise ValueError("training provenance decoder hash does not match model")

    blocks = [
        EvidenceBlockSpec(
            name="prior_z",
            role="prior",
            event_size=z_dim,
            event_reduction="sum",
            power=1.0,
            evidence_kind="log_prior",
            observation_support=f"R^{z_dim}, standard normal kernel",
        )
    ]
    if family == "demand":
        blocks.append(
            EvidenceBlockSpec(
                name="covariate_v",
                role="observation",
                event_size=v_dim,
                event_reduction="sum",
                power=1.0,
                evidence_kind="log_likelihood",
                observation_support=f"R^{v_dim}, isotropic Gaussian",
            )
        )
    else:
        vector_dim = int(model.vector_dim)
        blocks.extend(
            [
                EvidenceBlockSpec(
                    name="time",
                    role="observation",
                    event_size=1,
                    event_reduction="sum",
                    power=1.0,
                    evidence_kind="log_likelihood",
                    observation_support="R, Gaussian",
                ),
                EvidenceBlockSpec(
                    name="vector_proxy",
                    role="observation",
                    event_size=vector_dim,
                    event_reduction="sum",
                    power=1.0,
                    evidence_kind="log_likelihood",
                    observation_support=f"R^{vector_dim}, diagonal Gaussian",
                ),
            ]
        )
    spec = TargetSpec(
        family=family,
        blocks=tuple(blocks),
        global_power=float(global_power),
        decoder_model_hash=decoder_hash,
        preprocessor_hash=preprocessor.identity,
        model_training_provenance_hash=training_provenance.identity,
        use_bnn=bool(model.params.get("use_bnn", False)),
        dtype="float32",
    )
    return ResolvedTarget(model=model, spec=spec)


def _gaussian_event_log_score(observed, mean, variance):
    observed = tf.cast(observed, tf.float32)
    mean = tf.cast(mean, tf.float32)
    variance = tf.cast(variance, tf.float32)
    return -(
        tf.square(observed - mean) / (2.0 * variance)
        + 0.5 * tf.math.log(variance)
    )


@dataclass(frozen=True)
class ResolvedTarget:
    model: object
    spec: TargetSpec

    def validate_observation(self, data_v):
        """Validate shape/support/provenance once before target tracing."""
        data_v_np = np.asarray(data_v)
        if data_v_np.ndim != 2 or data_v_np.shape[1] != int(self.model.params["v_dim"]):
            raise ValueError("data_v shape does not match resolved target")
        try:
            finite_view = data_v_np.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("data_v must be numeric") from exc
        if not np.isfinite(finite_view).all():
            raise ValueError("data_v must be finite")

    def assert_runtime_identity(self):
        """Detect decoder mutation; called before and after every sampler pass."""
        if sha256_decoder(self.model) != self.spec.decoder_model_hash:
            raise RuntimeError("decoder model state changed after target resolution")

    def unpowered_event_sums(self, data_v, data_z):
        """Return base event sums before any block/global powers."""
        if tf.executing_eagerly():
            self.validate_observation(
                data_v.numpy() if tf.is_tensor(data_v) else data_v
            )
        data_v = tf.cast(data_v, tf.float32)
        data_z = tf.cast(data_z, tf.float32)
        result = {
            "prior_z": -tf.reduce_sum(tf.square(data_z), axis=1) / 2.0
        }
        if self.spec.family == "demand":
            output = self.model.g_net(data_z, training=False)
            mean = output[:, : int(self.model.params["v_dim"])]
            variance = self.model._continuous_sigma(
                output, sigma_key="sigma_v", eps=1e-6
            )
            result["covariate_v"] = tf.reduce_sum(
                _gaussian_event_log_score(data_v, mean, variance), axis=1
            )
            return result

        decoded = self.model._decode_covariates(data_z, training=False)
        time_observed = data_v[:, :1]
        result["time"] = tf.reduce_sum(
            _gaussian_event_log_score(
                time_observed, decoded["time_mean"], decoded["time_var"]
            ),
            axis=1,
        )
        if self.spec.family == "vector":
            vector_observed = data_v[:, 1 : 1 + int(self.model.vector_dim)]
            result["vector_proxy"] = tf.reduce_sum(
                _gaussian_event_log_score(
                    vector_observed,
                    decoded["vector_mean"],
                    decoded["vector_var"],
                ),
                axis=1,
            )
            return result
        raise ValueError("unsupported target family")

    def powered_blocks(self, data_v, data_z):
        base = self.unpowered_event_sums(data_v, data_z)
        return {
            block.name: tf.cast(block.power, tf.float32) * base[block.name]
            for block in self.spec.blocks
        }

    def log_prob(self, data_v, data_z):
        blocks = self.powered_blocks(data_v, data_z)
        # Same floating-point association as the training loss: observation
        # NLL blocks first, prior NLL second, then negated.
        observation_nll = tf.add_n(
            [
                -blocks[block.name]
                for block in self.spec.blocks
                if block.role == "observation"
            ]
        )
        prior_nll = -blocks["prior_z"]
        total = -(observation_nll + prior_nll)
        return tf.cast(self.spec.global_power, tf.float32) * total

    def log_prob_raw(self, raw_v, data_z, preprocessor: AffinePreprocessorSpec):
        """Evaluate raw covariates only when the hashed transform is exact."""
        if preprocessor.identity != self.spec.preprocessor_hash:
            raise ValueError("preprocessor hash does not match resolved target")
        return self.log_prob(preprocessor.transform(raw_v), data_z)


def independent_formula_oracle(model, data_v, data_z, *, global_power=1.0):
    """Direct formula for the target density, written independently of ``ResolvedTarget``."""
    family = _model_family(model)
    data_v = tf.cast(data_v, tf.float32)
    data_z = tf.cast(data_z, tf.float32)

    def gaussian_score(observed, mean, variance):
        # Repeated on purpose: the oracle must not share helpers with the target.
        observed = tf.cast(observed, tf.float32)
        mean = tf.cast(mean, tf.float32)
        variance = tf.cast(variance, tf.float32)
        return -0.5 * (
            tf.square(observed - mean) / variance + tf.math.log(variance)
        )

    prior = -tf.reduce_sum(tf.square(data_z), axis=1) / 2.0
    if family == "demand":
        output = model.g_net(data_z, training=False)
        mean = output[:, : int(model.params["v_dim"])]
        variance = model._continuous_sigma(output, sigma_key="sigma_v", eps=1e-6)
        covariance = tf.reduce_sum(
            gaussian_score(data_v, mean, variance), axis=1
        )
        return tf.cast(global_power, tf.float32) * (prior + covariance)

    decoded = model._decode_covariates(data_z, training=False)
    time = tf.reduce_sum(
        gaussian_score(
            data_v[:, :1], decoded["time_mean"], decoded["time_var"]
        ),
        axis=1,
    )
    if family == "vector":
        vector_events = gaussian_score(
            data_v[:, 1:], decoded["vector_mean"], decoded["vector_var"]
        )
        vector = tf.reduce_sum(vector_events, axis=1)
        return tf.cast(global_power, tf.float32) * (prior + time + vector)

    raise ValueError("unsupported target family")


__all__ = [
    "AffinePreprocessorSpec",
    "EvidenceBlockSpec",
    "FeaturePreprocessorSpec",
    "ModelTrainingProvenance",
    "PCAFeaturePreprocessorSpec",
    "ResolvedTarget",
    "TargetSpec",
    "independent_formula_oracle",
    "model_training_provenance",
    "require_digest",
    "resolve_target",
    "sha256_array",
    "sha256_decoder",
    "sha256_json",
    "sha256_weights",
]
