"""Central registry for root cause analyzers."""

from typing import Dict, Type, Any, Optional
import logging

logger = logging.getLogger(__name__)


class RCARegistry:
    """Registry for RCA analyzer discovery and instantiation."""
    
    _analyzers: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register an RCA analyzer class.
        
        Args:
            name: Unique identifier for the analyzer
        
        Returns:
            Decorator function
        """
        def decorator(analyzer_class: Type) -> Type:
            if name in cls._analyzers:
                logger.warning(f"Analyzer '{name}' is already registered. Overwriting.")
            
            cls._analyzers[name] = analyzer_class
            logger.debug(f"Registered RCA analyzer: {name} -> {analyzer_class.__name__}")
            return analyzer_class
        
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs: Any):
        """
        Factory method to create an analyzer instance.
        
        Args:
            name: Analyzer name (as registered)
            **kwargs: Keyword arguments to pass to analyzer constructor
        
        Returns:
            Analyzer instance
        
        Raises:
            ValueError: If analyzer name is not registered
        """
        if name not in cls._analyzers:
            available = ", ".join(cls.list())
            raise ValueError(
                f"Unknown RCA analyzer: '{name}'. Available analyzers: {available}"
            )
        
        analyzer_class = cls._analyzers[name]
        return analyzer_class(**kwargs)
    
    @classmethod
    def list(cls) -> list:
        """List all registered analyzer names."""
        return list(cls._analyzers.keys())
    
    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """Get analyzer class by name."""
        return cls._analyzers.get(name)
    
    @classmethod
    def clear(cls):
        """Clear all registered analyzers."""
        cls._analyzers.clear()
