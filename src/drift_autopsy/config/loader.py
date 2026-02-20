"""Configuration loader for YAML and JSON files."""

from pathlib import Path
from typing import Union, Dict, Any
import yaml
import json
import logging

from drift_autopsy.config.schema import PipelineConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Load and validate pipeline configurations from files or dictionaries.
    
    Supports YAML and JSON formats with Pydantic validation.
    """
    
    @staticmethod
    def from_yaml(path: Union[str, Path]) -> PipelineConfig:
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
        
        Returns:
            Validated PipelineConfig
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid or validation fails
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        logger.info(f"Loading configuration from: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            raise ValueError(f"Empty configuration file: {path}")
        
        try:
            config = PipelineConfig(**data)
            logger.info(f"Successfully loaded configuration: {config.name}")
            return config
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}") from e
    
    @staticmethod
    def from_json(path: Union[str, Path]) -> PipelineConfig:
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to JSON file
        
        Returns:
            Validated PipelineConfig
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid or validation fails
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        logger.info(f"Loading configuration from: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        try:
            config = PipelineConfig(**data)
            logger.info(f"Successfully loaded configuration: {config.name}")
            return config
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}") from e
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> PipelineConfig:
        """
        Load configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            Validated PipelineConfig
        
        Raises:
            ValueError: If validation fails
        """
        try:
            config = PipelineConfig(**config_dict)
            logger.info(f"Successfully loaded configuration: {config.name}")
            return config
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}") from e
    
    @staticmethod
    def to_yaml(config: PipelineConfig, path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: PipelineConfig to save
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict using Pydantic's model_dump
        config_dict = config.model_dump(exclude_none=True)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to: {path}")
    
    @staticmethod
    def to_json(config: PipelineConfig, path: Union[str, Path]) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config: PipelineConfig to save
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            f.write(config.model_dump_json(indent=2, exclude_none=True))
        
        logger.info(f"Configuration saved to: {path}")
