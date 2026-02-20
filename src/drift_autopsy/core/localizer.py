"""Base protocol for drift localizers."""

from typing import Protocol, Optional
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.result import DetectionResult, LocalizationResult


class DriftLocalizer(Protocol):
    """
    Protocol defining the interface for drift localizers.
    
    Localizers identify which features or data slices are affected by drift.
    """
    
    @property
    def name(self) -> str:
        """Return the name/identifier of the localizer."""
        ...
    
    def localize(
        self,
        reference_data: Dataset,
        test_data: Dataset,
        drift_signal: Optional[DetectionResult] = None,
    ) -> LocalizationResult:
        """
        Localize drift to specific features or regions.
        
        Args:
            reference_data: Reference dataset
            test_data: Test dataset
            drift_signal: Optional detection result from upstream detector
        
        Returns:
            LocalizationResult with feature-level drift information
        """
        ...


class BaseDriftLocalizer:
    """
    Base class for drift localizers providing common functionality.
    """
    
    def __init__(self, name: str):
        """
        Initialize base localizer.
        
        Args:
            name: Name/identifier for the localizer
        """
        self._name = name
    
    @property
    def name(self) -> str:
        """Return localizer name."""
        return self._name
    
    def localize(
        self,
        reference_data: Dataset,
        test_data: Dataset,
        drift_signal: Optional[DetectionResult] = None,
    ) -> LocalizationResult:
        """
        Localize drift.
        
        Args:
            reference_data: Reference dataset
            test_data: Test dataset
            drift_signal: Optional detection result
        
        Returns:
            LocalizationResult
        """
        # Subclasses should override this method
        raise NotImplementedError("Subclasses must implement localize()")
