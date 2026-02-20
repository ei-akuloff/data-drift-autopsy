"""Data validators for sanity checks."""

import logging
import numpy as np
import pandas as pd
from typing import Optional, List

from drift_autopsy.core.dataset import Dataset

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validate datasets for common issues.
    
    Checks for missing values, data types, shape consistency, etc.
    """
    
    @staticmethod
    def validate_dataset(
        dataset: Dataset,
        name: str = "dataset",
        check_missing: bool = True,
        check_inf: bool = True,
        check_variance: bool = True,
        min_samples: int = 10,
    ) -> None:
        """
        Validate a dataset and log warnings for issues.
        
        Args:
            dataset: Dataset to validate
            name: Name for logging
            check_missing: Check for missing values
            check_inf: Check for infinite values
            check_variance: Check for zero-variance features
            min_samples: Minimum number of samples required
        
        Raises:
            ValueError: If critical validation fails
        """
        logger.info(f"Validating {name}: shape={dataset.shape}")
        
        # Check minimum samples
        if dataset.n_samples < min_samples:
            raise ValueError(
                f"{name} has only {dataset.n_samples} samples, "
                f"minimum {min_samples} required"
            )
        
        # Convert to DataFrame for easier validation
        df = dataset.to_pandas()
        
        # Check for missing values
        if check_missing:
            missing_counts = df.isnull().sum()
            if missing_counts.any():
                missing_features = missing_counts[missing_counts > 0]
                logger.warning(
                    f"{name} has missing values in {len(missing_features)} features: "
                    f"{dict(missing_features.head())}"
                )
        
        # Check for infinite values
        if check_inf:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    logger.warning(
                        f"{name} has {inf_count} infinite values in feature '{col}'"
                    )
        
        # Check for zero-variance features
        if check_variance:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].std() == 0:
                    logger.warning(
                        f"{name} has zero variance in feature '{col}' (constant value)"
                    )
        
        logger.info(f"{name} validation complete")
    
    @staticmethod
    def validate_compatibility(
        reference: Dataset,
        test: Dataset,
        check_feature_names: bool = True,
        check_feature_order: bool = True,
    ) -> None:
        """
        Validate that two datasets are compatible for drift detection.
        
        Args:
            reference: Reference dataset
            test: Test dataset
            check_feature_names: Check feature names match
            check_feature_order: Check feature order matches
        
        Raises:
            ValueError: If datasets are incompatible
        """
        logger.info("Validating dataset compatibility")
        
        # Check same number of features
        if reference.n_features != test.n_features:
            raise ValueError(
                f"Feature count mismatch: reference has {reference.n_features}, "
                f"test has {test.n_features}"
            )
        
        # Check feature names match
        if check_feature_names:
            ref_features = set(reference.feature_names)
            test_features = set(test.feature_names)
            
            missing_in_test = ref_features - test_features
            extra_in_test = test_features - ref_features
            
            if missing_in_test:
                raise ValueError(
                    f"Features in reference but not in test: {missing_in_test}"
                )
            
            if extra_in_test:
                raise ValueError(
                    f"Features in test but not in reference: {extra_in_test}"
                )
        
        # Check feature order
        if check_feature_order:
            if reference.feature_names != test.feature_names:
                logger.warning(
                    "Feature order differs between reference and test. "
                    "This may affect some detectors."
                )
        
        logger.info("Dataset compatibility check passed")
