"""Base protocol for root cause analyzers."""

from typing import Protocol, Optional
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.result import LocalizationResult, RCAResult


class RootCauseAnalyzer(Protocol):
    """
    Protocol defining the interface for root cause analyzers.
    
    Analyzers investigate why drift occurred and provide explanations.
    """
    
    @property
    def name(self) -> str:
        """Return the name/identifier of the analyzer."""
        ...
    
    def analyze(
        self,
        reference_data: Dataset,
        test_data: Dataset,
        localization: Optional[LocalizationResult] = None,
        model: Optional[any] = None,
    ) -> RCAResult:
        """
        Analyze root causes of detected drift.
        
        Args:
            reference_data: Reference dataset
            test_data: Test dataset
            localization: Optional localization result from upstream
            model: Optional model for model-specific analysis (e.g., SHAP)
        
        Returns:
            RCAResult with explanations and recommendations
        """
        ...


class BaseRootCauseAnalyzer:
    """
    Base class for root cause analyzers providing common functionality.
    """
    
    def __init__(self, name: str):
        """
        Initialize base analyzer.
        
        Args:
            name: Name/identifier for the analyzer
        """
        self._name = name
    
    @property
    def name(self) -> str:
        """Return analyzer name."""
        return self._name
    
    def analyze(
        self,
        reference_data: Dataset,
        test_data: Dataset,
        localization: Optional[LocalizationResult] = None,
        model: Optional[any] = None,
    ) -> RCAResult:
        """
        Analyze root causes.
        
        Args:
            reference_data: Reference dataset
            test_data: Test dataset
            localization: Optional localization result
            model: Optional model
        
        Returns:
            RCAResult
        """
        # Subclasses should override this method
        raise NotImplementedError("Subclasses must implement analyze()")
