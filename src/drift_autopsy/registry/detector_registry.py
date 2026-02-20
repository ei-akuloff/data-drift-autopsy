"""Central registry for drift detectors with decorator-based registration."""

from typing import Dict, Type, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DetectorRegistry:
    """
    Central registry for detector discovery and instantiation.
    
    Supports decorator-based registration and factory pattern for creating detectors.
    """
    
    _detectors: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a detector class.
        
        Args:
            name: Unique identifier for the detector
        
        Returns:
            Decorator function
        
        Example:
            @DetectorRegistry.register("ks_test")
            class KSTest(BaseDriftDetector):
                ...
        """
        def decorator(detector_class: Type) -> Type:
            if name in cls._detectors:
                logger.warning(f"Detector '{name}' is already registered. Overwriting.")
            
            cls._detectors[name] = detector_class
            logger.debug(f"Registered detector: {name} -> {detector_class.__name__}")
            return detector_class
        
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs: Any):
        """
        Factory method to create a detector instance.
        
        Args:
            name: Detector name (as registered)
            **kwargs: Keyword arguments to pass to detector constructor
        
        Returns:
            Detector instance
        
        Raises:
            ValueError: If detector name is not registered
        
        Example:
            detector = DetectorRegistry.create("ks_test", threshold=0.05)
        """
        if name not in cls._detectors:
            available = ", ".join(cls.list())
            raise ValueError(
                f"Unknown detector: '{name}'. Available detectors: {available}"
            )
        
        detector_class = cls._detectors[name]
        return detector_class(**kwargs)
    
    @classmethod
    def list(cls) -> list:
        """
        List all registered detector names.
        
        Returns:
            List of detector names
        """
        return list(cls._detectors.keys())
    
    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """
        Get detector class by name.
        
        Args:
            name: Detector name
        
        Returns:
            Detector class or None if not found
        """
        return cls._detectors.get(name)
    
    @classmethod
    def clear(cls):
        """Clear all registered detectors (useful for testing)."""
        cls._detectors.clear()
