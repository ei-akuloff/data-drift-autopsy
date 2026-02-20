"""Maximum Mean Discrepancy (MMD) for drift detection."""

import numpy as np
from typing import Optional
import logging

from drift_autopsy.core.detector import BaseDriftDetector
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.result import DetectionResult, DriftSeverity
from drift_autopsy.registry import DetectorRegistry

logger = logging.getLogger(__name__)


@DetectorRegistry.register("mmd")
class MMD(BaseDriftDetector):
    """
    Maximum Mean Discrepancy for multivariate drift detection.
    
    MMD compares distributions by computing the distance between their
    mean embeddings in a reproducing kernel Hilbert space (RKHS).
    It captures multivariate shifts that univariate tests might miss.
    
    Args:
        threshold: MMD threshold for detecting drift (default: 0.1)
        kernel: Kernel type ("rbf" or "linear")
        gamma: RBF kernel parameter (default: None, auto-computed)
        n_permutations: Number of permutations for p-value computation (default: 100)
        max_samples: Maximum samples to use (subsamples if exceeded) for memory efficiency (default: 5000)
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        n_permutations: int = 100,
        max_samples: int = 5000,
    ):
        super().__init__(name="mmd")
        self.threshold = threshold
        self.kernel = kernel
        self.gamma = gamma
        self.n_permutations = n_permutations
        self.max_samples = max_samples
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
        """
        Compute RBF (Gaussian) kernel matrix.
        
        Args:
            X: First sample matrix (n_samples, n_features)
            Y: Second sample matrix (m_samples, n_features)
            gamma: Kernel bandwidth parameter
        
        Returns:
            Kernel matrix (n_samples, m_samples)
        """
        # Compute squared Euclidean distances
        XX = np.sum(X ** 2, axis=1).reshape(-1, 1)
        YY = np.sum(Y ** 2, axis=1).reshape(1, -1)
        XY = X @ Y.T
        
        sq_distances = XX + YY - 2 * XY
        
        # RBF kernel
        K = np.exp(-gamma * sq_distances)
        
        return K
    
    def _linear_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute linear kernel matrix."""
        return X @ Y.T
    
    def _subsample(self, X: np.ndarray, max_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Subsample data if it exceeds max_samples.
        
        Args:
            X: Data array (n_samples, n_features)
            max_samples: Maximum number of samples
            seed: Random seed for reproducibility
        
        Returns:
            Subsampled array
        """
        if len(X) <= max_samples:
            return X
        
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(X), size=max_samples, replace=False)
        return X[indices]
    
    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute MMD between two samples.
        
        Args:
            X: Reference sample (n, d)
            Y: Test sample (m, d)
        
        Returns:
            MMD value
        """
        n = len(X)
        m = len(Y)
        
        # Auto-compute gamma if not provided (median heuristic)
        gamma = self.gamma
        if gamma is None and self.kernel == "rbf":
            # Use median of pairwise distances
            combined = np.vstack([X[:min(100, n)], Y[:min(100, m)]])
            pairwise_sq_dists = np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=2)
            median_dist = np.median(pairwise_sq_dists[pairwise_sq_dists > 0])
            gamma = 1.0 / (2 * median_dist) if median_dist > 0 else 1.0
        
        # Compute kernels
        if self.kernel == "rbf":
            K_XX = self._rbf_kernel(X, X, gamma)
            K_YY = self._rbf_kernel(Y, Y, gamma)
            K_XY = self._rbf_kernel(X, Y, gamma)
        elif self.kernel == "linear":
            K_XX = self._linear_kernel(X, X)
            K_YY = self._linear_kernel(Y, Y)
            K_XY = self._linear_kernel(X, Y)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        
        # Compute MMD^2 (unbiased estimator)
        mmd_squared = (
            (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1))
            + (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))
            - 2 * K_XY.sum() / (n * m)
        )
        
        # MMD is the square root (but use squared for testing)
        mmd = np.sqrt(max(0, mmd_squared))
        
        return float(mmd)
    
    def _permutation_test(self, X: np.ndarray, Y: np.ndarray, mmd_obs: float) -> float:
        """
        Compute p-value using permutation test.
        
        Args:
            X: Reference sample
            Y: Test sample
            mmd_obs: Observed MMD
        
        Returns:
            P-value
        """
        n = len(X)
        combined = np.vstack([X, Y])
        
        null_mmds = []
        for _ in range(self.n_permutations):
            # Randomly permute
            perm_indices = np.random.permutation(len(combined))
            X_perm = combined[perm_indices[:n]]
            Y_perm = combined[perm_indices[n:]]
            
            # Compute MMD under null
            mmd_perm = self._compute_mmd(X_perm, Y_perm)
            null_mmds.append(mmd_perm)
        
        # P-value: proportion of permutations with MMD >= observed
        p_value = (np.sum(np.array(null_mmds) >= mmd_obs) + 1) / (self.n_permutations + 1)
        
        return float(p_value)
    
    def fit(self, reference_data: Dataset) -> None:
        """
        Store reference data.
        
        Args:
            reference_data: Reference dataset
        """
        super().fit(reference_data)
        logger.info(f"MMD fitted on {reference_data.n_samples} samples")
    
    def detect(self, test_data: Dataset) -> DetectionResult:
        """
        Detect drift using MMD.
        
        Args:
            test_data: Test dataset
        
        Returns:
            DetectionResult
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")
        
        logger.info(f"Running MMD on {test_data.n_samples} test samples")
        
        # Get numeric features as numpy arrays
        ref_df = self._reference_data.to_pandas()
        test_df = test_data.to_pandas()
        
        numeric_cols = ref_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric features found for MMD")
            return DetectionResult(
                detector_name=self.name,
                drift_detected=False,
                severity=DriftSeverity.NONE,
                score=0.0,
                threshold=self.threshold,
            )
        
        # Convert to numpy and handle missing values
        X = ref_df[numeric_cols].fillna(0).values
        Y = test_df[numeric_cols].fillna(0).values
        
        # Subsample if data is too large (for memory efficiency)
        original_n_ref = len(X)
        original_n_test = len(Y)
        
        X = self._subsample(X, self.max_samples, seed=42)
        Y = self._subsample(Y, self.max_samples, seed=42)
        
        if len(X) < original_n_ref or len(Y) < original_n_test:
            logger.info(
                f"Subsampled data for MMD: reference {original_n_ref} → {len(X)}, "
                f"test {original_n_test} → {len(Y)}"
            )
        
        # Compute MMD
        mmd_value = self._compute_mmd(X, Y)
        
        # Compute p-value if permutations enabled
        p_value = None
        if self.n_permutations > 0:
            p_value = self._permutation_test(X, Y, mmd_value)
            logger.debug(f"MMD permutation test p-value: {p_value:.4f}")
        
        # Determine drift
        drift_detected = mmd_value >= self.threshold
        
        # Determine severity
        if mmd_value < self.threshold:
            severity = DriftSeverity.NONE
        elif mmd_value < self.threshold * 1.5:
            severity = DriftSeverity.LOW
        elif mmd_value < self.threshold * 2.5:
            severity = DriftSeverity.MEDIUM
        elif mmd_value < self.threshold * 5:
            severity = DriftSeverity.HIGH
        else:
            severity = DriftSeverity.CRITICAL
        
        logger.info(
            f"MMD result: drift={drift_detected}, "
            f"mmd={mmd_value:.4f}, severity={severity.value}"
        )
        
        return DetectionResult(
            detector_name=self.name,
            drift_detected=drift_detected,
            severity=severity,
            score=mmd_value,
            threshold=self.threshold,
            p_value=p_value,
            statistic=mmd_value,
            metadata={
                "kernel": self.kernel,
                "n_features": len(numeric_cols),
                "n_permutations": self.n_permutations,
                "max_samples": self.max_samples,
                "subsampled": len(X) < original_n_ref or len(Y) < original_n_test,
                "actual_ref_samples": len(X),
                "actual_test_samples": len(Y),
            }
        )
