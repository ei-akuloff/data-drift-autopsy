"""Pipeline orchestration for drift analysis."""

import time
from typing import Optional, Union, Any
import logging

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.detector import DriftDetector
from drift_autopsy.core.localizer import DriftLocalizer
from drift_autopsy.core.rca import RootCauseAnalyzer
from drift_autopsy.core.result import PipelineResult, DetectionResult, LocalizationResult, RCAResult
from drift_autopsy.registry import DetectorRegistry, LocalizerRegistry, RCARegistry
from drift_autopsy.data.validators import DataValidator

logger = logging.getLogger(__name__)


class DriftPipeline:
    """
    Composable pipeline for drift detection, localization, and RCA.
    
    Orchestrates the full drift analysis workflow with conditional execution
    and proper error handling.
    
    Args:
        detector: Drift detector instance or name (for registry)
        localizer: Optional localizer instance or name
        rca: Optional RCA analyzer instance or name
        enable_localization: Enable localization step (default: True)
        enable_rca: Enable RCA step (default: False)
        validate_data: Validate input data (default: True)
        model: Optional model for RCA (e.g., for SHAP)
    
    Example:
        >>> from drift_autopsy import DriftPipeline
        >>> from drift_autopsy.detectors import KSTest
        >>> 
        >>> pipeline = DriftPipeline(
        ...     detector=KSTest(threshold=0.05),
        ...     localizer="univariate",
        ...     enable_rca=False
        ... )
        >>> result = pipeline.run(reference_data, test_data)
    """
    
    def __init__(
        self,
        detector: Union[DriftDetector, str],
        localizer: Optional[Union[DriftLocalizer, str]] = None,
        rca: Optional[Union[RootCauseAnalyzer, str]] = None,
        enable_localization: bool = True,
        enable_rca: bool = False,
        validate_data: bool = True,
        model: Optional[Any] = None,
    ):
        # Setup detector
        if isinstance(detector, str):
            self.detector = DetectorRegistry.create(detector)
            logger.info(f"Created detector from registry: {detector}")
        else:
            self.detector = detector
        
        # Setup localizer
        self.enable_localization = enable_localization and localizer is not None
        if self.enable_localization:
            if isinstance(localizer, str):
                self.localizer = LocalizerRegistry.create(localizer)
                logger.info(f"Created localizer from registry: {localizer}")
            else:
                self.localizer = localizer
        else:
            self.localizer = None
        
        # Setup RCA
        self.enable_rca = enable_rca and rca is not None
        if self.enable_rca:
            if isinstance(rca, str):
                self.rca = RCARegistry.create(rca)
                logger.info(f"Created RCA analyzer from registry: {rca}")
            else:
                self.rca = rca
        else:
            self.rca = None
        
        self.validate_data = validate_data
        self.model = model
        
        logger.info(
            f"Pipeline initialized: "
            f"detector={self.detector.name}, "
            f"localization={self.enable_localization}, "
            f"rca={self.enable_rca}"
        )
    
    def run(
        self,
        reference_data: Dataset,
        test_data: Dataset,
    ) -> PipelineResult:
        """
        Run the complete drift analysis pipeline.
        
        Args:
            reference_data: Reference dataset (e.g., training data)
            test_data: Test dataset (e.g., production data)
        
        Returns:
            PipelineResult with detection, localization, and RCA results
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("Starting drift analysis pipeline")
        logger.info(f"Reference: {reference_data.shape}, Test: {test_data.shape}")
        logger.info("=" * 60)
        
        # Validate data
        if self.validate_data:
            try:
                DataValidator.validate_dataset(reference_data, name="reference")
                DataValidator.validate_dataset(test_data, name="test")
                DataValidator.validate_compatibility(reference_data, test_data)
            except Exception as e:
                logger.error(f"Data validation failed: {e}")
                raise
        
        # Step 1: Drift Detection
        logger.info(f"[1/3] Running drift detection with {self.detector.name}")
        try:
            detection_result = self.detector.fit_detect(reference_data, test_data)
            logger.info(
                f"Detection complete: drift_detected={detection_result.drift_detected}, "
                f"severity={detection_result.severity.value}, "
                f"score={detection_result.score:.4f}"
            )
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            raise
        
        # Step 2: Drift Localization (conditional)
        localization_result = None
        if self.enable_localization:
            logger.info(f"[2/3] Running drift localization with {self.localizer.name}")
            try:
                localization_result = self.localizer.localize(
                    reference_data,
                    test_data,
                    drift_signal=detection_result,
                )
                logger.info(
                    f"Localization complete: "
                    f"{len(localization_result.drifted_features)} drifted features"
                )
            except Exception as e:
                logger.error(f"Drift localization failed: {e}")
                # Continue pipeline even if localization fails
                localization_result = None
        else:
            logger.info("[2/3] Localization disabled, skipping")
        
        # Step 3: Root Cause Analysis (conditional)
        rca_result = None
        if self.enable_rca:
            logger.info(f"[3/3] Running RCA with {self.rca.name}")
            try:
                rca_result = self.rca.analyze(
                    reference_data,
                    test_data,
                    localization=localization_result,
                    model=self.model,
                )
                logger.info("RCA complete")
            except Exception as e:
                logger.error(f"RCA failed: {e}")
                # Continue even if RCA fails
                rca_result = None
        else:
            logger.info("[3/3] RCA disabled, skipping")
        
        # Compute execution time
        execution_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info(f"Pipeline complete in {execution_time:.2f}s")
        logger.info("=" * 60)
        
        # Create pipeline result
        result = PipelineResult(
            detection=detection_result,
            localization=localization_result,
            rca=rca_result,
            execution_time_seconds=execution_time,
            metadata={
                "detector": self.detector.name,
                "localizer": self.localizer.name if self.localizer else None,
                "rca": self.rca.name if self.rca else None,
                "reference_samples": reference_data.n_samples,
                "test_samples": test_data.n_samples,
                "n_features": reference_data.n_features,
            }
        )
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        components = [f"detector={self.detector.name}"]
        if self.enable_localization:
            components.append(f"localizer={self.localizer.name}")
        if self.enable_rca:
            components.append(f"rca={self.rca.name}")
        
        return f"DriftPipeline({', '.join(components)})"
