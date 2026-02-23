"""
Hallucination Risk Detector — Confidence vs Density (Method 1).

Core Idea
---------
A model prediction is considered a *hallucination risk* when:

    high model confidence  AND  input far from training distribution

This is the "confident but likely wrong" regime that arises when a
model extrapolates outside its training manifold while producing
overconfident softmax probabilities.

Four interpretability quadrants
--------------------------------
                      │  Low Distance  │  High Distance
  ────────────────────┼────────────────┼──────────────────
  High Confidence     │  ✅ Safe       │  🚨 Hallucination
  Low  Confidence     │  😐 Uncertain  │  ⚠️  Honest UQ
  ────────────────────┴────────────────┴──────────────────

Density Methods
---------------
* "mahalanobis"       (default) — accounts for feature correlations,
                      research-grade signal, works best when d ≪ n.
* "knn"               — parameter-free, robust to non-Gaussian manifolds.
* "isolation_forest"  — handles high-dimensional data, tree-based anomaly.
* "kde"               — smooth density estimate, bandwidth-sensitive.

Composite Hallucination Score
------------------------------

    h(x) = confidence(x) * normalised_distance(x)

where normalised_distance = raw_distance / ref_95th_percentile,
clipped to [0, 1].

h(x) lies in [0, 1].  A sample is flagged when BOTH:

    confidence(x) >= confidence_threshold   (default 0.80)
    normalised_distance(x) >= distance_threshold  (default 0.50)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.detector import BaseDriftDetector
from drift_autopsy.core.result import DriftSeverity, HallucinationResult
from drift_autopsy.registry import DetectorRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity thresholds (fraction of test samples flagged)
# ---------------------------------------------------------------------------
_SEVERITY_THRESHOLDS = [
    (0.35, DriftSeverity.CRITICAL),
    (0.20, DriftSeverity.HIGH),
    (0.10, DriftSeverity.MEDIUM),
    (0.05, DriftSeverity.LOW),
    (0.00, DriftSeverity.NONE),
]

DensityMethod = Literal["mahalanobis", "knn", "isolation_forest", "kde"]


def _severity_from_rate(rate: float) -> DriftSeverity:
    for threshold, severity in _SEVERITY_THRESHOLDS:
        if rate >= threshold:
            return severity
    return DriftSeverity.NONE


# ---------------------------------------------------------------------------
# Public detector
# ---------------------------------------------------------------------------

@DetectorRegistry.register("hallucination_risk")
class HallucinationRiskDetector(BaseDriftDetector):
    """
    Detect per-sample hallucination risk using Confidence vs Density.

    The detector fits a density model on the **reference** feature matrix
    and, at inference time, flags test samples where the model is both
    *highly confident* and *far from the training distribution*.

    Parameters
    ----------
    density_method : str, default "mahalanobis"
        Density estimation backend.  One of:
        ``"mahalanobis"`` | ``"knn"`` | ``"isolation_forest"`` | ``"kde"``.
    confidence_threshold : float, default 0.80
        Minimum model confidence (max softmax probability) to be
        considered "high confidence".  Range (0, 1).
    distance_threshold : float, default 0.50
        Normalised distance cutoff above which a sample is considered
        "far from the training distribution".  Range (0, 1).
        (The raw distances are normalised by the reference 95th percentile.)
    distance_percentile : float, default 95.0
        Percentile of reference distances used to normalise test distances.
        Adjusting this trades sensitivity vs specificity.
    n_neighbors : int, default 5
        Number of nearest neighbours for ``density_method="knn"``.
    iso_n_estimators : int, default 200
        Number of trees for ``density_method="isolation_forest"``.
    kde_bandwidth : float or "scott", default "scott"
        Bandwidth for ``density_method="kde"``.
        Pass a float for a fixed bandwidth or ``"scott"`` for Scott's rule.
    random_state : int, optional
        Random seed for reproducible results (affects IsolationForest).

    Examples
    --------
    >>> from drift_autopsy import Dataset
    >>> from drift_autopsy.detectors.hallucination import HallucinationRiskDetector
    >>> detector = HallucinationRiskDetector(density_method="mahalanobis")
    >>> detector.fit(reference_dataset)
    >>> result = detector.detect(test_dataset)
    >>> print(result.hallucination_rate)
    >>> print(result.quadrant_counts)

    Notes
    -----
    * ``test_dataset`` **must** contain ``prediction_probabilities``
      (an (n, k) array of softmax probabilities) so that per-sample
      confidence can be extracted.
    * For binary classifiers storing only P(positive) as a 1-D array,
      confidence is computed as ``max(p, 1-p)``.
    """

    def __init__(
        self,
        density_method: DensityMethod = "mahalanobis",
        confidence_threshold: float = 0.80,
        distance_threshold: float = 0.50,
        distance_percentile: float = 95.0,
        n_neighbors: int = 5,
        iso_n_estimators: int = 200,
        kde_bandwidth: Any = "scott",
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(name="hallucination_risk")

        if density_method not in {"mahalanobis", "knn", "isolation_forest", "kde"}:
            raise ValueError(
                f"Unknown density_method '{density_method}'. "
                "Choose from: mahalanobis, knn, isolation_forest, kde"
            )

        self.density_method = density_method
        self.confidence_threshold = confidence_threshold
        self.distance_threshold = distance_threshold
        self.distance_percentile = distance_percentile
        self.n_neighbors = n_neighbors
        self.iso_n_estimators = iso_n_estimators
        self.kde_bandwidth = kde_bandwidth
        self.random_state = random_state

        # Fitted state
        self._ref_mean: Optional[np.ndarray] = None
        self._ref_VI: Optional[np.ndarray] = None          # mahalanobis
        self._nn_model = None                               # knn
        self._iso_model = None                              # isolation_forest
        self._kde_model = None                              # kde
        self._ref_distance_percentile: Optional[float] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, reference_data: Dataset) -> None:  # type: ignore[override]
        """
        Fit the density model on reference data.

        Parameters
        ----------
        reference_data : Dataset
            Training / reference distribution.  Only numeric features are
            used; categorical columns are silently dropped.
        """
        super().fit(reference_data)

        X_ref = self._extract_numeric(reference_data)
        logger.info(
            f"[HallucinationRiskDetector] Fitting '{self.density_method}' "
            f"on {X_ref.shape[0]} samples × {X_ref.shape[1]} features."
        )

        if self.density_method == "mahalanobis":
            self._fit_mahalanobis(X_ref)
        elif self.density_method == "knn":
            self._fit_knn(X_ref)
        elif self.density_method == "isolation_forest":
            self._fit_isolation_forest(X_ref)
        elif self.density_method == "kde":
            self._fit_kde(X_ref)

        # Compute normalisation constant from reference distances
        ref_distances = self._raw_distances(X_ref)
        self._ref_distance_percentile = float(
            np.percentile(ref_distances, self.distance_percentile)
        )
        logger.debug(
            f"[HallucinationRiskDetector] Reference {self.distance_percentile}th "
            f"percentile distance = {self._ref_distance_percentile:.4f}"
        )

    # ------------------------------------------------------------------
    # detect
    # ------------------------------------------------------------------

    def detect(self, test_data: Dataset) -> HallucinationResult:  # type: ignore[override]
        """
        Compute per-sample hallucination risk scores.

        Parameters
        ----------
        test_data : Dataset
            Must include ``prediction_probabilities``.

        Returns
        -------
        HallucinationResult
        """
        if not self._fitted:
            raise RuntimeError(
                "HallucinationRiskDetector must be fitted before calling detect()."
            )
        if test_data.prediction_probabilities is None:
            raise ValueError(
                "test_data.prediction_probabilities is required for hallucination "
                "detection.  Pass the model's softmax output when creating the Dataset."
            )

        X_test = self._extract_numeric(test_data)
        n = X_test.shape[0]

        logger.info(
            f"[HallucinationRiskDetector] Scoring {n} test samples "
            f"with method='{self.density_method}'."
        )

        # ---- 1. Confidence scores ----------------------------------------
        confidence_scores = self._extract_confidence(test_data)   # (n,) ∈ [0,1]

        # ---- 2. Density / distance scores ---------------------------------
        raw_distances = self._raw_distances(X_test)                # (n,) ≥ 0

        # Normalise: clip to [0, 1] using reference percentile
        norm_distances = np.clip(
            raw_distances / (self._ref_distance_percentile + 1e-10), 0.0, 1.0
        )  # (n,) ∈ [0,1]

        # ---- 3. Composite hallucination score -----------------------------
        hallucination_scores = confidence_scores * norm_distances   # (n,) ∈ [0,1]

        # ---- 4. Binary flag ------------------------------------------------
        high_confidence = confidence_scores >= self.confidence_threshold
        high_distance   = norm_distances    >= self.distance_threshold
        is_risk = high_confidence & high_distance                   # (n,) bool

        n_risk             = int(is_risk.sum())
        hallucination_rate = n_risk / n if n > 0 else 0.0
        severity           = _severity_from_rate(hallucination_rate)

        logger.info(
            f"[HallucinationRiskDetector] {n_risk}/{n} samples flagged "
            f"({hallucination_rate:.1%}) — severity={severity.value}"
        )

        metadata = {
            "density_method": self.density_method,
            "n_test_samples": n,
            "mean_confidence": float(confidence_scores.mean()),
            "mean_normalised_distance": float(norm_distances.mean()),
            "mean_hallucination_score": float(hallucination_scores.mean()),
            "ref_distance_percentile_value": self._ref_distance_percentile,
        }

        return HallucinationResult(
            detector_name=self.name,
            hallucination_scores=hallucination_scores,
            is_hallucination_risk=is_risk,
            confidence_scores=confidence_scores,
            density_scores=norm_distances,
            n_hallucination_risk=n_risk,
            hallucination_rate=hallucination_rate,
            severity=severity,
            confidence_threshold=self.confidence_threshold,
            distance_threshold=self.distance_threshold,
            metadata=metadata,
        )

    def fit_detect(                                     # type: ignore[override]
        self,
        reference_data: Dataset,
        test_data: Dataset,
    ) -> HallucinationResult:
        """Fit on reference then detect on test in one call."""
        self.fit(reference_data)
        return self.detect(test_data)

    # ------------------------------------------------------------------
    # Private — density backend fitting
    # ------------------------------------------------------------------

    def _fit_mahalanobis(self, X: np.ndarray) -> None:
        """
        Store mean and pseudo-inverse covariance matrix.

        Uses ``np.linalg.pinv`` for numerical stability when features are
        correlated or the covariance matrix is near-singular.
        """
        self._ref_mean = X.mean(axis=0)
        cov = np.cov(X.T)
        if cov.ndim == 0:
            # Single feature — treat as 1×1 matrix
            cov = np.array([[float(cov)]])
            self._ref_mean = np.array([self._ref_mean])
        self._ref_VI = np.linalg.pinv(cov)

    def _fit_knn(self, X: np.ndarray) -> None:
        """Fit k-Nearest Neighbours index on reference features."""
        from sklearn.neighbors import NearestNeighbors  # lazy import

        self._nn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm="auto",
            metric="euclidean",
            n_jobs=-1,
        )
        self._nn_model.fit(X)

    def _fit_isolation_forest(self, X: np.ndarray) -> None:
        """Fit Isolation Forest anomaly detector on reference features."""
        from sklearn.ensemble import IsolationForest  # lazy import

        self._iso_model = IsolationForest(
            n_estimators=self.iso_n_estimators,
            contamination="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._iso_model.fit(X)

    def _fit_kde(self, X: np.ndarray) -> None:
        """Fit Gaussian KDE on reference features."""
        from sklearn.neighbors import KernelDensity  # lazy import

        bandwidth = self.kde_bandwidth
        if bandwidth == "scott":
            # Scott's rule: n^(-1/(d+4))
            n, d = X.shape
            bandwidth = n ** (-1.0 / (d + 4))

        self._kde_model = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        self._kde_model.fit(X)

    # ------------------------------------------------------------------
    # Private — raw distance computation
    # ------------------------------------------------------------------

    def _raw_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute raw (unnormalised) density-based distances for each sample.

        Returns an (n,) array where higher values indicate samples that are
        farther from the reference distribution.
        """
        if self.density_method == "mahalanobis":
            return self._mahalanobis_distances(X)
        elif self.density_method == "knn":
            return self._knn_distances(X)
        elif self.density_method == "isolation_forest":
            return self._isolation_forest_distances(X)
        elif self.density_method == "kde":
            return self._kde_distances(X)
        else:
            raise RuntimeError(f"Unknown density_method: {self.density_method}")

    def _mahalanobis_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorised Mahalanobis distance.

        d(x) = sqrt( (x - μ)^T Σ^{-1} (x - μ) )

        Uses ``np.einsum`` for an efficient batched computation:
        einsum('ij,jk,ik->i', delta, VI, delta)
        """
        delta = X - self._ref_mean                    # (n, d)
        # einsum computes diag(delta @ VI @ delta.T) without full n×n matrix
        dist_sq = np.einsum("ij,jk,ik->i", delta, self._ref_VI, delta)
        return np.sqrt(np.maximum(dist_sq, 0.0))      # (n,)

    def _knn_distances(self, X: np.ndarray) -> np.ndarray:
        """Mean distance to k nearest neighbours in reference set."""
        distances, _ = self._nn_model.kneighbors(X)  # (n, k)
        return distances.mean(axis=1)                 # (n,)

    def _isolation_forest_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Anomaly score from Isolation Forest.

        ``score_samples`` returns lower values for anomalies.
        We negate so that higher output = farther from distribution.
        """
        return -self._iso_model.score_samples(X)      # (n,)

    def _kde_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Negative log-density as distance proxy.

        Higher value = lower density = farther from the distribution.
        """
        log_density = self._kde_model.score_samples(X)   # (n,)
        return -log_density                               # (n,)

    # ------------------------------------------------------------------
    # Private — helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_numeric(dataset: Dataset) -> np.ndarray:
        """
        Extract only numeric columns from a dataset as a float64 array.

        Missing values are imputed with the column mean.
        """
        df = dataset.to_pandas()
        numeric_df: pd.DataFrame = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError(
                "HallucinationRiskDetector requires at least one numeric feature."
            )
        # Mean-impute missing values
        numeric_df = numeric_df.fillna(numeric_df.mean())
        return numeric_df.values.astype(np.float64)

    @staticmethod
    def _extract_confidence(dataset: Dataset) -> np.ndarray:
        """
        Convert ``prediction_probabilities`` to a scalar confidence per sample.

        Handles three common shapes:

        * (n, k) — full softmax for k  ≥ 2 classes:
          confidence = max probability across classes.
        * (n,) or (n, 1) — binary P(positive):
          confidence = max(p, 1-p).

        Returns an (n,) float64 array in [0, 1].
        """
        proba = np.asarray(dataset.prediction_probabilities, dtype=np.float64)

        if proba.ndim == 1 or (proba.ndim == 2 and proba.shape[1] == 1):
            p = proba.ravel()
            # Binary: confidence = how decisive the prediction is
            confidence = np.maximum(p, 1.0 - p)
        elif proba.ndim == 2:
            # Multi-class: max softmax probability
            confidence = proba.max(axis=1)
        else:
            raise ValueError(
                f"prediction_probabilities has unexpected shape {proba.shape}. "
                "Expected (n,), (n, 1), or (n, k)."
            )

        return np.clip(confidence, 0.0, 1.0)
