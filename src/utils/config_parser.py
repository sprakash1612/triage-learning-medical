"""
Configuration parser for loading and managing YAML configurations.

Features:
- Load YAML configuration files
- Merge configurations (base + experiment specific)
- Validate configuration completeness
- Override with command-line arguments
- Save configurations to disk
"""

from typing import Any, Dict, List, Optional, Union
import yaml
import logging
import os
from pathlib import Path
from copy import deepcopy

logger = logging.getLogger(__name__)


class ConfigParser:
    """
    Parser for YAML configuration files with validation and merging capabilities.
    
    Provides a unified interface for loading, validating, merging, and saving
    configuration dictionaries from YAML files.
    """
    
    # Required top-level configuration keys
    REQUIRED_KEYS = {
        "model",
        "data",
        "training",
        "evaluation",
    }
    
    # Optional but recommended keys
    OPTIONAL_KEYS = {
        "uncertainty",
        "triage",
        "logging",
        "seed",
    }
    
    def __init__(
        self,
        base_config_path: Optional[str] = None,
        validate: bool = True,
    ):
        """
        Initialize ConfigParser.
        
        Args:
            base_config_path: Path to base configuration file
            validate: Whether to validate configuration upon loading
        """
        self.base_config_path = base_config_path
        self.config = {}
        self.validate = validate
        
        if base_config_path:
            self.load_config(base_config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        
        Returns:
            Loaded configuration dictionary
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If validation fails (if enabled)
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Loading configuration from {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {config_path}: {e}")
            raise
        
        if config is None:
            config = {}
        
        self.config = config
        
        if self.validate:
            self.validate_config(self.config)
        
        logger.info(f"Configuration loaded successfully from {config_path}")
        return self.config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Validate configuration completeness and structure.
        
        Args:
            config: Configuration dictionary to validate
        
        Raises:
            ValueError: If required keys are missing or validation fails
        """
        missing_keys = ConfigParser.REQUIRED_KEYS - set(config.keys())
        
        if missing_keys:
            raise ValueError(
                f"Configuration is missing required keys: {missing_keys}. "
                f"Required keys are: {ConfigParser.REQUIRED_KEYS}"
            )
        
        # Validate model config
        if "model" in config:
            if "architecture" not in config["model"]:
                raise ValueError("model.architecture is required")
            if "num_classes" not in config["model"]:
                raise ValueError("model.num_classes is required")
        
        # Validate data config
        if "data" in config:
            if "dataset" not in config["data"]:
                raise ValueError("data.dataset is required")
            if "batch_size" not in config["data"]:
                raise ValueError("data.batch_size is required")
        
        # Validate training config
        if "training" in config:
            if "epochs" not in config["training"]:
                raise ValueError("training.epochs is required")
            if "learning_rate" not in config["training"]:
                raise ValueError("training.learning_rate is required")
        
        logger.info("Configuration validation passed")
    
    def merge_configs(
        self,
        base_config: Union[str, Dict],
        experiment_config: Union[str, Dict],
        override_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Merge base configuration with experiment-specific configuration.
        
        Experiment config overrides base config. Supports deep merging for nested dicts.
        
        Args:
            base_config: Base configuration (file path or dict)
            experiment_config: Experiment configuration (file path or dict)
            override_keys: List of keys that should not be merged but replaced
        
        Returns:
            Merged configuration dictionary
        
        Example:
            >>> merged = parser.merge_configs('configs/base_config.yaml', 
            ...                                'configs/pathmnist_config.yaml')
        """
        # Load configs if paths are provided
        if isinstance(base_config, str):
            base = self.load_config(base_config)
        else:
            base = deepcopy(base_config)
        
        if isinstance(experiment_config, str):
            experiment = self.load_config(experiment_config)
        else:
            experiment = deepcopy(experiment_config)
        
        override_keys = override_keys or []
        
        logger.info("Merging configurations")
        
        # Deep merge
        merged = self._deep_merge(base, experiment, override_keys)
        
        return merged
    
    @staticmethod
    def _deep_merge(
        base: Dict,
        override: Dict,
        override_keys: Optional[List[str]] = None,
    ) -> Dict:
        """
        Recursively merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            override_keys: Keys that should be replaced, not merged
        
        Returns:
            Merged dictionary
        """
        override_keys = override_keys or []
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in override_keys:
                # Replace entire value for override keys
                result[key] = deepcopy(value)
            elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = ConfigParser._deep_merge(result[key], value, override_keys)
            else:
                # Override value
                result[key] = deepcopy(value)
        
        return result
    
    def override_config(
        self,
        config: Dict[str, Any],
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Override configuration with command-line or programmatic arguments.
        
        Supports dot notation for nested keys (e.g., 'model.architecture').
        
        Args:
            config: Configuration dictionary
            overrides: Dictionary of overrides with dot notation keys
        
        Returns:
            Updated configuration dictionary
        
        Example:
            >>> overrides = {'model.architecture': 'resnet50', 'training.epochs': 100}
            >>> updated = parser.override_config(config, overrides)
        """
        result = deepcopy(config)
        
        for key_path, value in overrides.items():
            keys = key_path.split('.')
            target = result
            
            # Navigate to the parent of the final key
            for key in keys[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]
            
            # Set the final value
            final_key = keys[-1]
            target[final_key] = value
            
            logger.debug(f"Overriding {key_path} = {value}")
        
        logger.info(f"Applied {len(overrides)} configuration overrides")
        return result
    
    def save_config(
        self,
        config: Dict[str, Any],
        output_path: str,
        include_metadata: bool = True,
    ) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary to save
            output_path: Path to save configuration file
            include_metadata: Whether to include metadata comments
        
        Raises:
            IOError: If file cannot be written
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                if include_metadata:
                    f.write("# Configuration file\n")
                    f.write("# Auto-generated configuration\n\n")
                
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration saved to {output_path}")
        except IOError as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_nested(
        self,
        config: Dict[str, Any],
        key_path: str,
        default: Any = None,
    ) -> Any:
        """
        Get nested configuration value using dot notation.
        
        Args:
            config: Configuration dictionary
            key_path: Dot-notation path (e.g., 'model.architecture')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        
        Example:
            >>> arch = parser.get_nested(config, 'model.architecture', 'resnet50')
        """
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_nested(
        self,
        config: Dict[str, Any],
        key_path: str,
        value: Any,
    ) -> Dict[str, Any]:
        """
        Set nested configuration value using dot notation.
        
        Args:
            config: Configuration dictionary
            key_path: Dot-notation path (e.g., 'model.architecture')
            value: Value to set
        
        Returns:
            Updated configuration dictionary
        """
        keys = key_path.split('.')
        target = config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
        return config
    
    @staticmethod
    def dict_to_argparse_args(config: Dict[str, Any]) -> List[str]:
        """
        Convert configuration dictionary to command-line arguments.
        
        Useful for passing config to external scripts.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            List of command-line arguments in format ['--key', 'value', ...]
        """
        args = []
        for key, value in config.items():
            args.append(f"--{key}")
            args.append(str(value))
        return args
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigParser(config_keys={list(self.config.keys())})"
