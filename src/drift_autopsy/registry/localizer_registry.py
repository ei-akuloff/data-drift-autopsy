"""Central registry for drift localizers."""

from typing import Dict, Type, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LocalizerRegistry:
    """Registry for localizer discovery and instantiation."""
    
    _localizers: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a localizer class.
        
        Args:
            name: Unique identifier for the localizer
        
        Returns:
            Decorator function
        """
        def decorator(localizer_class: Type) -> Type:
            if name in cls._localizers:
                logger.warning(f"Localizer '{name}' is already registered. Overwriting.")
            
            cls._localizers[name] = localizer_class
            logger.debug(f"Registered localizer: {name} -> {localizer_class.__name__}")
            return localizer_class
        
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs: Any):
        """
        Factory method to create a localizer instance.
        
        Args:
            name: Localizer name (as registered)
            **kwargs: Keyword arguments to pass to localizer constructor
        
        Returns:
            Localizer instance
        
        Raises:
            ValueError: If localizer name is not registered
        """
        if name not in cls._localizers:
            available = ", ".join(cls.list())
            raise ValueError(
                f"Unknown localizer: '{name}'. Available localizers: {available}"
            )
        
        localizer_class = cls._localizers[name]
        return localizer_class(**kwargs)
    
    @classmethod
    def list(cls) -> list:
        """List all registered localizer names."""
        return list(cls._localizers.keys())
    
    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """Get localizer class by name."""
        return cls._localizers.get(name)
    
    @classmethod
    def clear(cls):
        """Clear all registered localizers."""
        cls._localizers.clear()
