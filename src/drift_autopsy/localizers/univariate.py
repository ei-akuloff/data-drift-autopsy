"""Univariate drift localization."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, List, Dict, Any
import logging

from drift_autopsy.core.localizer import BaseDriftLocalizer
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.result import DetectionResult, LocalizationResult, FeatureDrift, DriftSeverity
from drift_autopsy.registry import LocalizerRegistry

logger = logging.getLogger(__name__)


@LocalizerRegistry.register("univariate")
class UnivariateLocalizer(BaseDriftLocalizer):
    """
    Univariate drift localization using per-feature statistical tests.
    
    Runs statistical tests (KS test for continuous, chi-square for categorical)
    on each feature individually to identify which features are drifting.
    
    Args:
        threshold: P-value threshold for feature-level drift (default: 0.05)
        correction: Multiple testing correction ("bonferroni", "holm", or None)
        top_k: Return only top k drifted features (default: None = all)
        min_samples_categorical: Minimum samples for categorical test (default: 5)
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        correction: Optional[str] = "bonferroni",
        top_k: Optional[int] = None,
        min_samples_categorical: int = 5,
    ):
        super().__init__(name="univariate")
        self.threshold = threshold
        self.correction = correction
        self.top_k = top_k
        self.min_samples_categorical = min_samples_categorical
    
    def _test_numeric_feature(
        self,
        ref_values: np.ndarray,
        test_values: np.ndarray,
        feature_name: str,
    ) -> tuple:
        """
        Test numeric feature using KS test.
        
        Returns:
            (statistic, p_value, distribution_info)
        """
        statistic, p_value = stats.ks_2samp(ref_values, test_values)
        
        # Compute distribution statistics
        dist_info = {
            "ref_mean": float(np.mean(ref_values)),
            "ref_std": float(np.std(ref_values)),
            "test_mean": float(np.mean(test_values)),
            "test_std": float(np.std(test_values)),
            "mean_shift": float(np.mean(test_values) - np.mean(ref_values)),
            "std_shift": float(np.std(test_values) - np.std(ref_values)),
        }
        
        return float(statistic), float(p_value), dist_info
    
    def _test_categorical_feature(
        self,
        ref_values: pd.Series,
        test_values: pd.Series,
        feature_name: str,
    ) -> tuple:
        """
        Test categorical feature using chi-square test.
        
        Returns:
            (statistic, p_value, distribution_info)
        """
        # Get value counts
        ref_counts = ref_values.value_counts()
        test_counts = test_values.value_counts()
        
        # Align categories
        all_categories = sorted(set(ref_counts.index) | set(test_counts.index))
        
        ref_counts_aligned = np.array([ref_counts.get(cat, 0) for cat in all_categories])
        test_counts_aligned = np.array([test_counts.get(cat, 0) for cat in all_categories])
        
        # Filter categories with minimum samples
        valid_mask = (ref_counts_aligned >= self.min_samples_categorical) | \
                     (test_counts_aligned >= self.min_samples_categorical)
        
        if valid_mask.sum() < 2:
            # Not enough categories for test
            return 0.0, 1.0, {"message": "Insufficient categories"}
        
        ref_counts_valid = ref_counts_aligned[valid_mask]
        test_counts_valid = test_counts_aligned[valid_mask]
        
        # Normalize to get expected proportions
        ref_proportions = ref_counts_valid / ref_counts_valid.sum()
        expected_counts = ref_proportions * test_counts_valid.sum()
        
        # Chi-square test
        try:
            chi2_stat, p_value = stats.chisquare(test_counts_valid, expected_counts)
            
            dist_info = {
                "n_categories": len(all_categories),
                "n_tested_categories": valid_mask.sum(),
                "ref_mode": ref_values.mode()[0] if len(ref_values.mode()) > 0 else None,
                "test_mode": test_values.mode()[0] if len(test_values.mode()) > 0 else None,
            }
            
            return float(chi2_stat), float(p_value), dist_info
        except Exception as e:
            logger.warning(f"Chi-square test failed for '{feature_name}': {e}")
            return 0.0, 1.0, {"error": str(e)}
    
    def localize(
        self,
        reference_data: Dataset,
        test_data: Dataset,
        drift_signal: Optional[DetectionResult] = None,
    ) -> LocalizationResult:
        """
        Localize drift to individual features.
        
        Args:
            reference_data: Reference dataset
            test_data: Test dataset
            drift_signal: Optional upstream drift detection result
        
        Returns:
            LocalizationResult with per-feature drift information
        """
        logger.info("Running univariate localization")
        
        ref_df = reference_data.to_pandas()
        test_df = test_data.to_pandas()
        
        feature_drifts = []
        
        # Test numeric features
        numeric_cols = ref_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in test_df.columns:
                logger.warning(f"Feature '{col}' not in test data, skipping")
                continue
            
            ref_values = ref_df[col].dropna().values
            test_values = test_df[col].dropna().values
            
            if len(ref_values) == 0 or len(test_values) == 0:
                logger.warning(f"Skipping '{col}' due to insufficient data")
                continue
            
            statistic, p_value, dist_info = self._test_numeric_feature(
                ref_values, test_values, col
            )
            
            feature_drifts.append({
                "name": col,
                "type": "numeric",
                "statistic": statistic,
                "p_value": p_value,
                "dist_info": dist_info,
            })
        
        # Test categorical features
        categorical_cols = ref_df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in test_df.columns:
                continue
            
            ref_values = ref_df[col].dropna()
            test_values = test_df[col].dropna()
            
            if len(ref_values) == 0 or len(test_values) == 0:
                continue
            
            statistic, p_value, dist_info = self._test_categorical_feature(
                ref_values, test_values, col
            )
            
            feature_drifts.append({
                "name": col,
                "type": "categorical",
                "statistic": statistic,
                "p_value": p_value,
                "dist_info": dist_info,
            })
        
        # Apply multiple testing correction
        if len(feature_drifts) == 0:
            logger.warning("No features could be tested")
            return LocalizationResult(
                method_name=self.name,
                feature_drifts=[],
            )
        
        # Extract p-values
        p_values = np.array([fd["p_value"] for fd in feature_drifts])
        
        # Adjust threshold
        adjusted_threshold = self.threshold
        if self.correction == "bonferroni":
            adjusted_threshold = self.threshold / len(p_values)
            logger.debug(f"Bonferroni correction: adjusted threshold = {adjusted_threshold:.6f}")
        elif self.correction == "holm":
            # Sort p-values and apply Holm correction
            sorted_indices = np.argsort(p_values)
            adjusted_thresholds = self.threshold / (len(p_values) - np.arange(len(p_values)))
        
        # Create FeatureDrift objects
        feature_drift_objects = []
        
        for i, fd in enumerate(feature_drifts):
            if self.correction == "holm":
                idx_in_sorted = np.where(sorted_indices == i)[0][0]
                threshold_for_feature = adjusted_thresholds[idx_in_sorted]
            else:
                threshold_for_feature = adjusted_threshold
            
            drift_detected = fd["p_value"] < threshold_for_feature
            
            # Determine severity
            if fd["p_value"] >= threshold_for_feature:
                severity = DriftSeverity.NONE
            elif fd["p_value"] >= threshold_for_feature / 2:
                severity = DriftSeverity.LOW
            elif fd["p_value"] >= threshold_for_feature / 10:
                severity = DriftSeverity.MEDIUM
            elif fd["p_value"] >= threshold_for_feature / 100:
                severity = DriftSeverity.HIGH
            else:
                severity = DriftSeverity.CRITICAL
            
            feature_drift_objects.append(
                FeatureDrift(
                    feature_name=fd["name"],
                    drift_detected=drift_detected,
                    score=fd["statistic"],
                    p_value=fd["p_value"],
                    severity=severity,
                    distribution_shift=fd["dist_info"],
                )
            )
        
        # Sort by p-value (most drifted first)
        feature_drift_objects.sort(key=lambda x: x.p_value)
        
        # Limit to top_k if specified
        if self.top_k is not None:
            feature_drift_objects = feature_drift_objects[:self.top_k]
        
        n_drifted = sum(1 for fd in feature_drift_objects if fd.drift_detected)
        
        logger.info(
            f"Localization complete: {n_drifted}/{len(feature_drift_objects)} features drifted"
        )
        
        return LocalizationResult(
            method_name=self.name,
            feature_drifts=feature_drift_objects,
            metadata={
                "threshold": adjusted_threshold,
                "correction": self.correction,
                "n_tested": len(feature_drifts),
                "n_drifted": n_drifted,
            }
        )
