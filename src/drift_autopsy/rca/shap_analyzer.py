"""SHAP-based root cause analysis."""

import numpy as np
from typing import Optional, Any, Dict, List
import logging

# NumPy 2.0 compatibility shims
if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
    np.trapz = np.trapezoid

if not hasattr(np, 'in1d'):
    np.in1d = np.isin

from drift_autopsy.core.rca import BaseRootCauseAnalyzer
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.result import LocalizationResult, RCAResult
from drift_autopsy.registry import RCARegistry

logger = logging.getLogger(__name__)


@RCARegistry.register("shap")
class SHAPAnalyzer(BaseRootCauseAnalyzer):
    """
    SHAP-based root cause analysis for drift.
    
    Computes SHAP values on reference and test data to understand:
    1. How feature importances have changed
    2. Which features contribute most to predictions
    3. How the model's reliance on features has shifted
    
    Requires:
        - A trained model that is compatible with SHAP
        - Reference and test data
    
    Args:
        n_background_samples: Number of samples for SHAP background (default: 100)
        n_test_samples: Number of test samples to explain (default: 100)
        feature_subset: Optional list of features to focus on
    """
    
    def __init__(
        self,
        n_background_samples: int = 100,
        n_test_samples: int = 100,
        feature_subset: Optional[List[str]] = None,
    ):
        super().__init__(name="shap")
        self.n_background_samples = n_background_samples
        self.n_test_samples = n_test_samples
        self.feature_subset = feature_subset
    
    def analyze(
        self,
        reference_data: Dataset,
        test_data: Dataset,
        localization: Optional[LocalizationResult] = None,
        model: Optional[Any] = None,
    ) -> RCAResult:
        """
        Analyze root causes using SHAP.
        
        Args:
            reference_data: Reference dataset
            test_data: Test dataset
            localization: Optional localization result
            model: Trained model for SHAP analysis
        
        Returns:
            RCAResult with SHAP-based explanations
        """
        if model is None:
            logger.warning("No model provided for SHAP analysis. Skipping.")
            return RCAResult(
                analyzer_name=self.name,
                explanations={"error": "Model required for SHAP analysis"},
                recommendations=["Provide a trained model to enable SHAP analysis"],
            )
        
        try:
            import shap
        except ImportError:
            logger.error("SHAP library not installed")
            return RCAResult(
                analyzer_name=self.name,
                explanations={"error": "SHAP library not installed"},
                recommendations=["Install SHAP: pip install shap"],
            )
        
        logger.info("Running SHAP analysis")
        
        # Sample data for efficiency
        ref_df = reference_data.to_pandas()
        test_df = test_data.to_pandas()
        
        # Get numeric features
        numeric_cols = ref_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return RCAResult(
                analyzer_name=self.name,
                explanations={"error": "No numeric features for SHAP"},
            )
        
        # Determine which features to focus analysis on (for reporting/recommendations)
        # But we'll compute SHAP on ALL features so model gets correct input shape
        if self.feature_subset:
            features_to_focus = [f for f in self.feature_subset if f in numeric_cols]
        elif localization and localization.drifted_features:
            # Focus analysis on drifted features
            features_to_focus = [f for f in localization.drifted_features if f in numeric_cols]
        else:
            features_to_focus = numeric_cols
        
        if len(features_to_focus) == 0:
            features_to_focus = numeric_cols  # Fallback
        
        logger.info(f"Analyzing {len(features_to_focus)} features with SHAP")
        
        # Prepare data - use ALL numeric features so model gets correct input shape
        X_ref = ref_df[numeric_cols].fillna(0).values
        X_test = test_df[numeric_cols].fillna(0).values
        
        # Sample for efficiency
        n_ref = min(self.n_background_samples, len(X_ref))
        n_test = min(self.n_test_samples, len(X_test))
        
        ref_indices = np.random.choice(len(X_ref), n_ref, replace=False)
        test_indices = np.random.choice(len(X_test), n_test, replace=False)
        
        X_ref_sample = X_ref[ref_indices]
        X_test_sample = X_test[test_indices]
        
        try:
            # Create explainer
            explainer = shap.Explainer(model.predict, X_ref_sample)
            
            # Compute SHAP values for reference
            logger.debug("Computing SHAP values for reference data")
            shap_values_ref = explainer(X_ref_sample)
            
            # Compute SHAP values for test
            logger.debug("Computing SHAP values for test data")
            shap_values_test = explainer(X_test_sample)
            
            # Extract values (handle different SHAP return types)
            if hasattr(shap_values_ref, 'values'):
                shap_ref = shap_values_ref.values
                shap_test = shap_values_test.values
            else:
                shap_ref = shap_values_ref
                shap_test = shap_values_test
            
            # Handle multi-class (take absolute mean across classes)
            if len(shap_ref.shape) == 3:
                shap_ref = np.abs(shap_ref).mean(axis=2)
                shap_test = np.abs(shap_test).mean(axis=2)
            
            # Compute feature importances (mean absolute SHAP)
            ref_importance = np.abs(shap_ref).mean(axis=0)
            test_importance = np.abs(shap_test).mean(axis=0)
            
            importance_shift = test_importance - ref_importance
            
            # Create importance dictionaries for ALL features
            feature_importances = {}
            importance_changes = {}
            
            for i, feature_name in enumerate(numeric_cols):
                feature_importances[feature_name] = {
                    "ref_importance": float(ref_importance[i]),
                    "test_importance": float(test_importance[i]),
                    "change": float(importance_shift[i]),
                    "relative_change": float(
                        importance_shift[i] / (ref_importance[i] + 1e-10)
                    ),
                }
                importance_changes[feature_name] = float(importance_shift[i])
            
            # Sort by absolute change, but prioritize features_to_focus
            # First, focus features sorted by change
            focus_changes = {f: importance_changes[f] for f in features_to_focus if f in importance_changes}
            sorted_focus = sorted(focus_changes.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Then, other features sorted by change
            other_changes = {f: c for f, c in importance_changes.items() if f not in features_to_focus}
            sorted_other = sorted(other_changes.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Combined: focus features first, then others
            sorted_changes = sorted_focus + sorted_other
            
            # Generate recommendations based on focused features
            recommendations = []
            
            # Top focused features with increased importance
            increased = [(f, c) for f, c in sorted_focus if c > 0][:3]
            if increased:
                recommendations.append(
                    f"Features with increased model reliance: {', '.join([f for f, _ in increased])}"
                )
            
            # Top focused features with decreased importance
            decreased = [(f, c) for f, c in sorted_focus if c < 0][:3]
            if decreased:
                recommendations.append(
                    f"Features with decreased model reliance: {', '.join([f for f, _ in decreased])}"
                )
            
            # Combine with localization insights
            if localization and localization.drifted_features:
                drifted_and_important = set(localization.drifted_features) & set([f for f, _ in sorted_changes[:5]])
                if drifted_and_important:
                    recommendations.append(
                        f"Features both drifted and importance-shifted: {', '.join(drifted_and_important)} - likely root causes"
                    )
            
            logger.info("SHAP analysis complete")
            
            return RCAResult(
                analyzer_name=self.name,
                explanations={
                    "method": "SHAP feature importance comparison",
                    "n_reference_samples": n_ref,
                    "n_test_samples": n_test,
                    "top_importance_changes": sorted_changes[:10],
                },
                feature_importances=importance_changes,
                distribution_changes=feature_importances,
                recommendations=recommendations,
                metadata={
                    "n_features_analyzed": len(numeric_cols),
                    "n_features_focused": len(features_to_focus),
                    "features_focused": features_to_focus,
                }
            )
        
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            return RCAResult(
                analyzer_name=self.name,
                explanations={"error": f"SHAP analysis failed: {str(e)}"},
                recommendations=["Check model compatibility with SHAP", "Verify data format"],
            )
