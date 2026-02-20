"""Population Stability Index (PSI) for drift detection."""

import numpy as np
import pandas as pd
from typing import Optional
import logging

from drift_autopsy.core.detector import BaseDriftDetector
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.result import DetectionResult, DriftSeverity
from drift_autopsy.registry import DetectorRegistry

logger = logging.getLogger(__name__)


@DetectorRegistry.register("psi")
class PSI(BaseDriftDetector):
    """
    Population Stability Index for detecting distribution drift.
    
    PSI measures the shift in distributions by binning data and comparing
    the proportions in each bin. It works for both continuous and categorical features.
    
    PSI = sum((actual% - expected%) * ln(actual% / expected%))
    
    Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change
    - PSI >= 0.2: Significant change
    
    Args:
        threshold: PSI threshold for detecting drift (default: 0.2)
        n_bins: Number of bins for continuous features (default: 10)
        min_bin_size: Minimum proportion in a bin to avoid division issues (default: 0.001)
        aggregate_method: How to aggregate PSI across features ("max" or "mean")
    """
    
    def __init__(
        self,
        threshold: float = 0.2,
        n_bins: int = 10,
        min_bin_size: float = 0.001,
        aggregate_method: str = "max",
    ):
        super().__init__(name="psi")
        self.threshold = threshold
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.aggregate_method = aggregate_method
        self._bin_edges = {}
    
    def fit(self, reference_data: Dataset) -> None:
        """
        Learn binning scheme from reference data.
        
        Args:
            reference_data: Reference dataset
        """
        super().fit(reference_data)
        
        ref_df = self._reference_data.to_pandas()
        
        # For each numeric feature, compute quantile-based bins
        numeric_cols = ref_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            values = ref_df[col].dropna()
            if len(values) > 0:
                # Use quantile-based binning to ensure balanced bins
                try:
                    _, bin_edges = pd.qcut(
                        values,
                        q=self.n_bins,
                        retbins=True,
                        duplicates='drop'
                    )
                    self._bin_edges[col] = bin_edges
                except Exception as e:
                    logger.warning(f"Could not create bins for feature '{col}': {e}")
                    # Fallback to uniform bins
                    bin_edges = np.linspace(values.min(), values.max(), self.n_bins + 1)
                    self._bin_edges[col] = bin_edges
        
        logger.info(f"PSI fitted on {reference_data.n_samples} samples with {len(self._bin_edges)} binned features")
    
    def _calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
    ) -> float:
        """
        Calculate PSI between two distributions.
        
        Args:
            expected: Expected (reference) proportions
            actual: Actual (test) proportions
        
        Returns:
            PSI value
        """
        # Add small constant to avoid division by zero
        expected = np.maximum(expected, self.min_bin_size)
        actual = np.maximum(actual, self.min_bin_size)
        
        # Normalize to sum to 1
        expected = expected / expected.sum()
        actual = actual / actual.sum()
        
        # Calculate PSI
        psi = np.sum((actual - expected) * np.log(actual / expected))
        
        return float(psi)
    
    def detect(self, test_data: Dataset) -> DetectionResult:
        """
        Detect drift using PSI for each feature.
        
        Args:
            test_data: Test dataset
        
        Returns:
            DetectionResult with aggregated PSI
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")
        
        logger.info(f"Running PSI on {test_data.n_samples} test samples")
        
        ref_df = self._reference_data.to_pandas()
        test_df = test_data.to_pandas()
        
        psi_values = []
        feature_results = {}
        
        # Calculate PSI for each binned feature
        for col, bin_edges in self._bin_edges.items():
            if col not in test_df.columns:
                logger.warning(f"Feature '{col}' not in test data, skipping")
                continue
            
            ref_values = ref_df[col].dropna()
            test_values = test_df[col].dropna()
            
            if len(ref_values) == 0 or len(test_values) == 0:
                logger.warning(f"Skipping feature '{col}' due to insufficient data")
                continue
            
            # Bin the data
            ref_binned = pd.cut(ref_values, bins=bin_edges, include_lowest=True)
            test_binned = pd.cut(test_values, bins=bin_edges, include_lowest=True)
            
            # Count proportions
            ref_counts = ref_binned.value_counts(normalize=True, sort=False)
            test_counts = test_binned.value_counts(normalize=True, sort=False)
            
            # Align indices (in case some bins are empty in test)
            ref_counts, test_counts = ref_counts.align(test_counts, fill_value=self.min_bin_size)
            
            # Calculate PSI
            psi = self._calculate_psi(ref_counts.values, test_counts.values)
            psi_values.append(psi)
            
            feature_results[col] = {
                "psi": float(psi),
                "n_bins": len(bin_edges) - 1,
            }
        
        # Check categorical features
        categorical_cols = ref_df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in test_df.columns:
                continue
            
            ref_values = ref_df[col].dropna()
            test_values = test_df[col].dropna()
            
            if len(ref_values) == 0 or len(test_values) == 0:
                continue
            
            # Get value counts
            ref_counts = ref_values.value_counts(normalize=True)
            test_counts = test_values.value_counts(normalize=True)
            
            # Align
            ref_counts, test_counts = ref_counts.align(test_counts, fill_value=self.min_bin_size)
            
            # Calculate PSI
            psi = self._calculate_psi(ref_counts.values, test_counts.values)
            psi_values.append(psi)
            
            feature_results[col] = {
                "psi": float(psi),
                "n_categories": len(ref_counts),
            }
        
        if len(psi_values) == 0:
            logger.warning("No features could be tested")
            return DetectionResult(
                detector_name=self.name,
                drift_detected=False,
                severity=DriftSeverity.NONE,
                score=0.0,
                threshold=self.threshold,
            )
        
        # Aggregate PSI values
        if self.aggregate_method == "max":
            aggregated_psi = float(np.max(psi_values))
        elif self.aggregate_method == "mean":
            aggregated_psi = float(np.mean(psi_values))
        else:
            aggregated_psi = float(np.max(psi_values))
        
        # Determine drift and severity
        drift_detected = aggregated_psi >= self.threshold
        
        if aggregated_psi < 0.1:
            severity = DriftSeverity.NONE
        elif aggregated_psi < 0.2:
            severity = DriftSeverity.LOW
        elif aggregated_psi < 0.3:
            severity = DriftSeverity.MEDIUM
        elif aggregated_psi < 0.5:
            severity = DriftSeverity.HIGH
        else:
            severity = DriftSeverity.CRITICAL
        
        logger.info(
            f"PSI result: drift={drift_detected}, "
            f"psi={aggregated_psi:.4f}, severity={severity.value}"
        )
        
        return DetectionResult(
            detector_name=self.name,
            drift_detected=drift_detected,
            severity=severity,
            score=aggregated_psi,
            threshold=self.threshold,
            metadata={
                "n_features_tested": len(psi_values),
                "feature_results": feature_results,
                "aggregate_method": self.aggregate_method,
            }
        )
