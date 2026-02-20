"""Base protocol for drift detectors."""

from typing import Protocol, Optional
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.result import DetectionResult


class DriftDetector(Protocol):
    """
    Protocol defining the interface for drift detectors.
    
    All drift detectors should implement this interface to ensure
    compatibility with the pipeline and registry system.
    """
    
    @property
    def name(self) -> str:
        """Return the name/identifier of the detector."""
        ...
    
    def fit(self, reference_data: Dataset) -> None:
        """
        Learn from reference distribution.
        
        Args:
            reference_data: Reference dataset (e.g., training data)
        """
        ...
    
    def detect(self, test_data: Dataset) -> DetectionResult:
        """
        Detect drift in test data relative to reference.
        
        Args:
            test_data: Test dataset to check for drift
        
        Returns:
            DetectionResult containing drift detection outcome
        """
        ...
    
    def fit_detect(self, reference_data: Dataset, test_data: Dataset) -> DetectionResult:
        """
        Convenience method to fit and detect in one call.
        
        Args:
            reference_data: Reference dataset
            test_data: Test dataset
        
        Returns:
            DetectionResult containing drift detection outcome
        """
        ...


class BaseDriftDetector:
    """
    Base class for drift detectors providing common functionality.
    
    Detectors can inherit from this class or implement the DriftDetector protocol directly.
    """
    
    def __init__(self, name: str):
        """
        Initialize base detector.
        
        Args:
            name: Name/identifier for the detector
        """
        self._name = name
        self._fitted = False
        self._reference_data: Optional[Dataset] = None
    
    @property
    def name(self) -> str:
        """Return detector name."""
        return self._name
    
    @property
    def fitted(self) -> bool:
        """Check if detector has been fitted."""
        return self._fitted
    
    def fit(self, reference_data: Dataset) -> None:
        """
        Fit detector on reference data.
        
        Args:
            reference_data: Reference dataset
        """
        self._reference_data = reference_data
        self._fitted = True
    
    def detect(self, test_data: Dataset) -> DetectionResult:
        """
        Detect drift in test data.
        
        Args:
            test_data: Test dataset
        
        Returns:
            DetectionResult
        
        Raises:
            RuntimeError: If detector has not been fitted
        """
        if not self._fitted:
            raise RuntimeError(f"Detector '{self.name}' must be fitted before calling detect()")
        
        # Subclasses should override this method
        raise NotImplementedError("Subclasses must implement detect()")
    
    def fit_detect(self, reference_data: Dataset, test_data: Dataset) -> DetectionResult:
        """
        Fit and detect in one call.
        
        Args:
            reference_data: Reference dataset
            test_data: Test dataset
        
        Returns:
            DetectionResult
        """
        self.fit(reference_data)
        return self.detect(test_data)
