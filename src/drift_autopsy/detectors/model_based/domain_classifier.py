"""Domain classifier for drift detection."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from typing import Optional
import logging

from drift_autopsy.core.detector import BaseDriftDetector
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.result import DetectionResult, DriftSeverity
from drift_autopsy.registry import DetectorRegistry

logger = logging.getLogger(__name__)


@DetectorRegistry.register("domain_classifier")
class DomainClassifier(BaseDriftDetector):
    """
    Domain classifier for drift detection.
    
    Trains a binary classifier to distinguish between reference and test data.
    High AUC (>> 0.5) indicates the distributions are distinguishable, i.e., drift.
    
    This method is powerful for detecting multivariate shifts where features
    interact in complex ways.
    
    Args:
        threshold: AUC threshold for detecting drift (default: 0.6)
        n_estimators: Number of trees in random forest (default: 100)
        max_depth: Maximum depth of trees (default: 5)
        use_cross_val: Use cross-validation for AUC (default: True)
        cv_folds: Number of CV folds (default: 3)
    """
    
    def __init__(
        self,
        threshold: float = 0.6,
        n_estimators: int = 100,
        max_depth: int = 5,
        use_cross_val: bool = True,
        cv_folds: int = 3,
        random_state: int = 42,
    ):
        super().__init__(name="domain_classifier")
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.use_cross_val = use_cross_val
        self.cv_folds = cv_folds
        self.random_state = random_state
        self._classifier = None
    
    def fit(self, reference_data: Dataset) -> None:
        """
        Store reference data.
        
        Args:
            reference_data: Reference dataset
        """
        super().fit(reference_data)
        logger.info(f"Domain classifier fitted on {reference_data.n_samples} reference samples")
    
    def detect(self, test_data: Dataset) -> DetectionResult:
        """
        Detect drift using domain classifier.
        
        Args:
            test_data: Test dataset
        
        Returns:
            DetectionResult with AUC as drift score
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")
        
        logger.info(f"Running domain classifier on {test_data.n_samples} test samples")
        
        # Get numeric features
        ref_df = self._reference_data.to_pandas()
        test_df = test_data.to_pandas()
        
        numeric_cols = ref_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric features found for domain classifier")
            return DetectionResult(
                detector_name=self.name,
                drift_detected=False,
                severity=DriftSeverity.NONE,
                score=0.5,
                threshold=self.threshold,
            )
        
        # Prepare data: reference = 0, test = 1
        X_ref = ref_df[numeric_cols].fillna(0).values
        X_test = test_df[numeric_cols].fillna(0).values
        
        X = np.vstack([X_ref, X_test])
        y = np.hstack([
            np.zeros(len(X_ref)),
            np.ones(len(X_test))
        ])
        
        # Train classifier
        self._classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        if self.use_cross_val:
            # Use cross-validation for more robust estimate
            try:
                cv_scores = cross_val_score(
                    self._classifier,
                    X,
                    y,
                    cv=self.cv_folds,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                auc = float(np.mean(cv_scores))
                auc_std = float(np.std(cv_scores))
                logger.debug(f"Cross-validated AUC: {auc:.4f} ± {auc_std:.4f}")
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}. Using train AUC.")
                self._classifier.fit(X, y)
                y_pred_proba = self._classifier.predict_proba(X)[:, 1]
                auc = roc_auc_score(y, y_pred_proba)
                auc_std = None
        else:
            # Simple train AUC (may overestimate)
            self._classifier.fit(X, y)
            y_pred_proba = self._classifier.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_pred_proba)
            auc_std = None
        
        # Get feature importances
        if not self.use_cross_val or self._classifier is None:
            self._classifier.fit(X, y)
        
        feature_importances = dict(
            zip(numeric_cols, self._classifier.feature_importances_)
        )
        
        # Sort by importance
        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10
        
        # Determine drift
        drift_detected = auc >= self.threshold
        
        # Determine severity based on how distinguishable the distributions are
        # AUC = 0.5: no drift, AUC = 1.0: complete separation
        if auc < 0.55:
            severity = DriftSeverity.NONE
        elif auc < 0.65:
            severity = DriftSeverity.LOW
        elif auc < 0.75:
            severity = DriftSeverity.MEDIUM
        elif auc < 0.85:
            severity = DriftSeverity.HIGH
        else:
            severity = DriftSeverity.CRITICAL
        
        logger.info(
            f"Domain classifier result: drift={drift_detected}, "
            f"AUC={auc:.4f}, severity={severity.value}"
        )
        
        return DetectionResult(
            detector_name=self.name,
            drift_detected=drift_detected,
            severity=severity,
            score=auc,
            threshold=self.threshold,
            statistic=auc,
            metadata={
                "auc": auc,
                "auc_std": auc_std,
                "n_features": len(numeric_cols),
                "top_features": sorted_features,
                "use_cross_val": self.use_cross_val,
            }
        )
