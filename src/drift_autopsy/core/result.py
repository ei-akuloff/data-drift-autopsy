"""Result dataclasses for drift detection and analysis."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import numpy as np


class DriftSeverity(Enum):
    """Severity levels for detected drift."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectionResult:
    """Result of drift detection."""
    
    detector_name: str
    drift_detected: bool
    severity: DriftSeverity
    score: float
    threshold: float
    p_value: Optional[float] = None
    statistic: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detector_name": self.detector_name,
            "drift_detected": self.drift_detected,
            "severity": self.severity.value,
            "score": self.score,
            "threshold": self.threshold,
            "p_value": self.p_value,
            "statistic": self.statistic,
            "metadata": self.metadata,
        }


@dataclass
class FeatureDrift:
    """Drift information for a single feature."""
    
    feature_name: str
    drift_detected: bool
    score: float
    p_value: Optional[float] = None
    severity: DriftSeverity = DriftSeverity.NONE
    distribution_shift: Optional[Dict[str, Any]] = None


@dataclass
class LocalizationResult:
    """Result of drift localization."""
    
    method_name: str
    feature_drifts: List[FeatureDrift]
    drifted_features: List[str] = field(default_factory=list)
    drift_scores: Dict[str, float] = field(default_factory=dict)
    
    # Slice-based localization
    slice_drifts: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Extract drifted features for convenience."""
        self.drifted_features = [
            fd.feature_name for fd in self.feature_drifts if fd.drift_detected
        ]
        self.drift_scores = {
            fd.feature_name: fd.score for fd in self.feature_drifts
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method_name": self.method_name,
            "drifted_features": self.drifted_features,
            "drift_scores": self.drift_scores,
            "feature_drifts": [
                {
                    "feature_name": fd.feature_name,
                    "drift_detected": fd.drift_detected,
                    "score": fd.score,
                    "p_value": fd.p_value,
                    "severity": fd.severity.value,
                }
                for fd in self.feature_drifts
            ],
            "slice_drifts": self.slice_drifts,
            "metadata": self.metadata,
        }


@dataclass
class RCAResult:
    """Result of root cause analysis."""
    
    analyzer_name: str
    explanations: Dict[str, Any]
    feature_importances: Optional[Dict[str, float]] = None
    distribution_changes: Optional[Dict[str, Any]] = None
    visualizations: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analyzer_name": self.analyzer_name,
            "explanations": self.explanations,
            "feature_importances": self.feature_importances,
            "distribution_changes": self.distribution_changes,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


@dataclass
class PipelineResult:
    """Complete result from drift analysis pipeline."""
    
    detection: DetectionResult
    localization: Optional[LocalizationResult] = None
    rca: Optional[RCAResult] = None
    execution_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detection": self.detection.to_dict(),
            "localization": self.localization.to_dict() if self.localization else None,
            "rca": self.rca.to_dict() if self.rca else None,
            "execution_time_seconds": self.execution_time_seconds,
            "metadata": self.metadata,
        }


@dataclass
class HallucinationResult:
    """
    Per-sample result from hallucination risk detection.

    A sample is flagged as hallucination-risk when the model is
    simultaneously *high-confidence* and *far from the training
    distribution* — the canonical "confident but likely wrong" regime.

    Attributes:
        detector_name:          Name of the density method used.
        hallucination_scores:   Per-sample composite risk score in [0, 1].
                                score = confidence * normalised_distance.
        is_hallucination_risk:  Boolean mask — True where both
                                confidence > confidence_threshold AND
                                normalised_distance > distance_threshold.
        confidence_scores:      Raw per-sample confidence (max softmax prob
                                or 1 - predictive entropy).
        density_scores:         Raw per-sample distance from training
                                distribution (already normalised to [0, 1]
                                using the reference 95th-percentile).
        n_hallucination_risk:   Number of flagged samples.
        hallucination_rate:     Fraction of test samples flagged in [0, 1].
        severity:               Aggregate severity derived from rate.
        confidence_threshold:   Threshold applied to confidence scores.
        distance_threshold:     Normalised distance threshold applied.
        metadata:               Extra method-specific diagnostics.
    """

    detector_name: str
    hallucination_scores: np.ndarray
    is_hallucination_risk: np.ndarray
    confidence_scores: np.ndarray
    density_scores: np.ndarray
    n_hallucination_risk: int
    hallucination_rate: float
    severity: "DriftSeverity"
    confidence_threshold: float
    distance_threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serialisable dictionary."""
        return {
            "detector_name": self.detector_name,
            "n_hallucination_risk": self.n_hallucination_risk,
            "hallucination_rate": float(self.hallucination_rate),
            "severity": self.severity.value,
            "confidence_threshold": self.confidence_threshold,
            "distance_threshold": self.distance_threshold,
            "hallucination_scores": self.hallucination_scores.tolist(),
            "is_hallucination_risk": self.is_hallucination_risk.tolist(),
            "confidence_scores": self.confidence_scores.tolist(),
            "density_scores": self.density_scores.tolist(),
            "metadata": self.metadata,
        }

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------

    @property
    def flagged_indices(self) -> np.ndarray:
        """Indices of samples flagged as hallucination-risk."""
        return np.where(self.is_hallucination_risk)[0]

    @property
    def quadrant_counts(self) -> Dict[str, int]:
        """
        Break the test set into the four interpretability quadrants.

        Returns:
            {
              "safe":               low distance + high confidence,
              "uncertain_honest":   high distance + low confidence,
              "hallucination_risk": high distance + high confidence,
              "uncertain_safe":     low distance + low confidence,
            }
        """
        high_conf = self.confidence_scores >= self.confidence_threshold
        high_dist = self.density_scores >= self.distance_threshold
        return {
            "safe":               int(np.sum(~high_dist & high_conf)),
            "uncertain_honest":   int(np.sum(high_dist & ~high_conf)),
            "hallucination_risk": int(np.sum(high_dist & high_conf)),
            "uncertain_safe":     int(np.sum(~high_dist & ~high_conf)),
        }
