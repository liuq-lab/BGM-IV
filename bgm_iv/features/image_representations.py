"""Image-representation export and preprocessing for MNIST covariates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np
from scipy.cluster.vq import kmeans2
import tensorflow as tf

from ..hashing import sha256_array, sha256_weights


def unique_rows(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First-occurrence unique rows (byte-exact) and the inverse index."""

    order: dict[bytes, int] = {}
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
    return np.stack(rows, axis=0), inverse


def image_encoder_features(trunk, raw_v, *, feature_slice: slice = slice(1, 65)) -> np.ndarray:
    """Inference-mode trunk output restricted to the image-feature slice.

    ONE forward pass over the byte-unique rows of the array, scattered back to
    every row: identical raw rows therefore get bit-identical features on any
    device (float32 convolution reductions are not batch-position invariant,
    so a row-wise pass could split one catalog target into two).
    """

    raw = np.asarray(raw_v, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError("raw_v must be [n, v_dim]")
    unique, inverse = unique_rows(raw)
    output = trunk(tf.constant(unique), training=False)
    phi_unique = np.asarray(output, np.float32)[:, feature_slice]
    expected = int(feature_slice.stop - feature_slice.start)
    if phi_unique.shape[1] != expected:
        raise ValueError(f"trunk feature width {phi_unique.shape[1]} != {expected}")
    if not np.all(np.isfinite(phi_unique)):
        raise ValueError("trunk features are non-finite")
    return np.ascontiguousarray(phi_unique[inverse])


def fit_feature_standardizer(train_phi) -> dict[str, np.ndarray]:
    phi = np.asarray(train_phi, np.float64)
    if phi.ndim != 2 or phi.shape[0] < 2:
        raise ValueError("train_phi must be [n >= 2, d]")
    mean = phi.mean(axis=0)
    scale = phi.std(axis=0, ddof=1) + 1e-6
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}


def standardize_features(phi, stats: Mapping[str, np.ndarray]) -> np.ndarray:
    phi = np.asarray(phi, np.float32)
    return ((phi - stats["mean"][None, :]) / stats["scale"][None, :]).astype(np.float32)


def fit_pca_control(train_pixels, k: int = 64) -> dict[str, np.ndarray]:
    """PCA-k on pixels/255 of the n TRAINING images (fairness control)."""

    images = np.asarray(train_pixels, np.float64) / 255.0
    if images.ndim != 2 or images.shape[0] <= k:
        raise ValueError("need more training images than PCA components")
    pca_mean = images.mean(axis=0)
    _, _, vt = np.linalg.svd(images - pca_mean, full_matrices=False)
    return {
        "components": vt[:k].astype(np.float32),
        "mean": pca_mean.astype(np.float32),
    }


def pca_features(pixels, pca: Mapping[str, np.ndarray]) -> np.ndarray:
    images = np.asarray(pixels, np.float32) / 255.0
    return ((images - pca["mean"][None, :]) @ pca["components"].T).astype(np.float32)


def within_cluster_floor_rule(
    train_std,
    holdout_std,
    *,
    factor: float = 0.5,
    minimum: float = 0.02,
    clusters: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Label-free floor: k-means centroids fitted on TRAIN features, residuals
    scored on HELD-OUT rows assigned to their nearest centroid.

    Returns ``max(minimum, factor x median within-cluster residual sd)`` with
    the training-side diagnostics (cluster occupancy, residual sd quantiles,
    PCA k90/k99 of the held-out features).  Evaluation-grid rows must never
    be passed here.
    """

    feat = np.asarray(train_std, np.float64)
    feat_ho = np.asarray(holdout_std, np.float64)
    clusters = int(clusters)
    if feat.ndim != 2 or feat_ho.ndim != 2 or feat.shape[1] != feat_ho.shape[1]:
        raise ValueError("train and held-out features must be [n, d] with equal d")
    if clusters < 1 or feat.shape[0] < clusters:
        raise ValueError("clusters must be positive and at most the number of training rows")
    centroids, _ = kmeans2(feat, clusters, minit="++", seed=int(seed))
    distances = ((feat_ho[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    assignment = np.argmin(distances, axis=1)
    occupancy = np.bincount(assignment, minlength=clusters)
    resid = feat_ho - centroids[assignment]
    within_sd = resid.std(0, ddof=1)
    centred = feat_ho - feat_ho.mean(0)
    spectrum = np.linalg.svd(centred, compute_uv=False) ** 2
    cumulative = np.cumsum(spectrum) / spectrum.sum()
    median_within = float(np.median(within_sd))
    value = float(max(float(minimum), round(float(factor) * median_within, 3)))
    return {
        "sigma_vector_softfloor": value,
        "rule": f"{factor} x median within-cluster residual sd (standardized units), min {minimum}",
        "rule_factor": float(factor),
        "rule_minimum": float(minimum),
        "clusters": clusters,
        "cluster_seed": int(seed),
        "holdout_cluster_occupancy_min": int(occupancy.min()),
        "within_cluster_resid_sd_median": median_within,
        "within_cluster_resid_sd_p10": float(np.quantile(within_sd, 0.1)),
        "within_cluster_resid_sd_min": float(within_sd.min()),
        "pca_k90": int(np.searchsorted(cumulative, 0.90) + 1),
        "pca_k99": int(np.searchsorted(cumulative, 0.99) + 1),
        "feature_dim_sd_min": float(feat.std(0).min()),
        "n_train": int(feat.shape[0]),
        "n_holdout": int(feat_ho.shape[0]),
    }


def build_feature_covariates(time_column, phi_std, noise_block=None) -> np.ndarray:
    """Assemble ``v~ = (time, standardized phi[, noise])`` as float32."""

    parts = [np.asarray(time_column, np.float32).reshape(-1, 1), np.asarray(phi_std, np.float32)]
    if noise_block is not None:
        parts.append(np.asarray(noise_block, np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32)


@dataclass
class ImageRepresentationExport:
    """Processed image covariates and their provenance."""

    feature_map: str
    feature_dim: int
    pixel_v_dim: int
    stats: dict
    train_v: np.ndarray
    grid_v: np.ndarray
    holdout_v: np.ndarray
    floor: dict
    trunk_weights_sha256: Optional[str]
    trunk_num_params: Optional[int]
    pca: Optional[dict] = None
    hashes: dict = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "feature_map": self.feature_map,
            "feature_dim": int(self.feature_dim),
            "pixel_v_dim": int(self.pixel_v_dim),
            "noise_dim": int(self.train_v.shape[1] - 1 - self.feature_dim),
            "trunk_weights_sha256": self.trunk_weights_sha256,
            "trunk_num_params": self.trunk_num_params,
            "standardizer_mean_sha256": sha256_array(self.stats["mean"]),
            "standardizer_scale_sha256": sha256_array(self.stats["scale"]),
            "floor": dict(self.floor),
            "hashes": dict(self.hashes),
        }


def export_image_representations(
    *,
    trunk,
    train: Mapping[str, np.ndarray],
    grid: Mapping[str, np.ndarray],
    holdout: Mapping[str, np.ndarray],
    pixel_v_dim: int,
    feature_map: str = "egm",
    feature_dim: int = 64,
    floor_factor: float = 0.5,
    floor_minimum: float = 0.02,
    pca_components: int = 64,
) -> ImageRepresentationExport:
    """Build learned or PCA image representations for one run.

    ``train``/``grid``/``holdout`` carry raw pixel covariates ``v``; no labels
    are used.  For ``pixel_v_dim > 785`` the raw nuisance block ``v[:, 785:]``
    is passed through as a third covariate block.
    """

    pixel_v_dim = int(pixel_v_dim)
    feature_slice = slice(1, 1 + int(feature_dim))
    for name, part in (("train", train), ("grid", grid), ("holdout", holdout)):
        if np.asarray(part["v"]).shape[1] != pixel_v_dim:
            raise ValueError(f"{name} covariates are not {pixel_v_dim}-wide")
    if feature_map == "egm":
        phi_tr = image_encoder_features(trunk, train["v"], feature_slice=feature_slice)
        phi_gr = image_encoder_features(trunk, grid["v"], feature_slice=feature_slice)
        phi_ho = image_encoder_features(trunk, holdout["v"], feature_slice=feature_slice)
        trunk_hash = sha256_weights(trunk)
        trunk_params = int(sum(int(np.prod(w.shape)) for w in trunk.weights))
        pca = None
    elif feature_map == "pca":
        pca = fit_pca_control(np.asarray(train["v"])[:, 1:785], k=int(pca_components))
        phi_tr = pca_features(np.asarray(train["v"])[:, 1:785], pca)
        phi_gr = pca_features(np.asarray(grid["v"])[:, 1:785], pca)
        phi_ho = pca_features(np.asarray(holdout["v"])[:, 1:785], pca)
        trunk_hash, trunk_params = None, None
    else:
        raise ValueError("feature_map must be 'egm' or 'pca'")
    stats = fit_feature_standardizer(phi_tr)
    std_tr, std_gr, std_ho = (standardize_features(p, stats) for p in (phi_tr, phi_gr, phi_ho))
    floor = within_cluster_floor_rule(
        std_tr, std_ho, factor=floor_factor, minimum=floor_minimum
    )
    noise = (lambda part: np.asarray(part["v"], np.float32)[:, 785:]) if pixel_v_dim > 785 else (lambda part: None)
    train_v = build_feature_covariates(np.asarray(train["v"])[:, :1], std_tr, noise(train))
    grid_v = build_feature_covariates(np.asarray(grid["v"])[:, :1], std_gr, noise(grid))
    holdout_v = build_feature_covariates(np.asarray(holdout["v"])[:, :1], std_ho, noise(holdout))
    hashes = {
        "phi_train_sha256": sha256_array(phi_tr),
        "phi_grid_sha256": sha256_array(phi_gr),
        "phi_holdout_sha256": sha256_array(phi_ho),
        "vtilde_train_sha256": sha256_array(train_v),
        "vtilde_grid_sha256": sha256_array(grid_v),
        "vtilde_holdout_sha256": sha256_array(holdout_v),
    }
    if pca is not None:
        hashes["pca_components_sha256"] = sha256_array(pca["components"])
    return ImageRepresentationExport(
        feature_map=str(feature_map),
        feature_dim=int(feature_dim),
        pixel_v_dim=pixel_v_dim,
        stats=stats,
        train_v=train_v,
        grid_v=grid_v,
        holdout_v=holdout_v,
        floor=floor,
        trunk_weights_sha256=trunk_hash,
        trunk_num_params=trunk_params,
        pca=pca,
        hashes=hashes,
    )


__all__ = [
    "ImageRepresentationExport",
    "sha256_array",
    "build_feature_covariates",
    "export_image_representations",
    "fit_feature_standardizer",
    "fit_pca_control",
    "image_encoder_features",
    "pca_features",
    "standardize_features",
    "within_cluster_floor_rule",
]
