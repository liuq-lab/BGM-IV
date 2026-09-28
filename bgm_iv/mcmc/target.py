from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class AffinePreprocessorSpec:
    mean: np.ndarray
    scale: np.ndarray

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

    def transform(self, raw_v):
        raw = np.asarray(raw_v, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != self.dimension:
            raise ValueError("raw_v shape does not match preprocessor dimension")
        mean = np.asarray(self.mean, dtype=np.float32).reshape(1, -1)
        scale = np.asarray(self.scale, dtype=np.float32).reshape(1, -1)
        return ((raw - mean) / scale).astype(np.float32)


def _model_family(model) -> str:
    name = type(model).__name__
    if name == "BGM_IV_Image":
        return "mnist"
    if name == "BGM_IV":
        return "demand"
    raise TypeError(f"unsupported model class: {name}")


def _gaussian_event_log_score(observed, mean, variance):
    observed = tf.cast(observed, tf.float32)
    mean = tf.cast(mean, tf.float32)
    variance = tf.cast(variance, tf.float32)
    return -(
        tf.square(observed - mean) / (2.0 * variance)
        + 0.5 * tf.math.log(variance)
    )


def latent_log_prob(model, family: str, data_v, data_z):
    data_v = tf.cast(data_v, tf.float32)
    data_z = tf.cast(data_z, tf.float32)
    prior = -tf.reduce_sum(tf.square(data_z), axis=1) / 2.0
    if family == "demand":
        output = model.g_net(data_z, training=False)
        mean = output[:, : int(model.params["v_dim"])]
        variance = model._continuous_sigma(output, sigma_key="sigma_v", eps=1e-6)
        covariate = tf.reduce_sum(
            _gaussian_event_log_score(data_v, mean, variance), axis=1
        )
        observation_nll = tf.add_n([-covariate])
    else:
        decoded = model._decode_covariates(data_z, training=False)
        time = tf.reduce_sum(
            _gaussian_event_log_score(
                data_v[:, :1], decoded["time_mean"], decoded["time_var"]
            ),
            axis=1,
        )
        pixels = tf.reduce_sum(
            -tf.nn.sigmoid_cross_entropy_with_logits(
                labels=data_v[:, 1:785] / 255.0,
                logits=decoded["image_logits"],
            ),
            axis=1,
        )
        observation_nll = tf.add_n([-time, -pixels])
    prior_nll = -prior
    return -(observation_nll + prior_nll)


__all__ = [
    "AffinePreprocessorSpec",
    "latent_log_prob",
]
