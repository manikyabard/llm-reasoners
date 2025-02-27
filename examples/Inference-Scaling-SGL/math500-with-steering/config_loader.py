"""
Configuration loader for math500-with-steering experiments.
This module provides functionality to load and validate YAML configuration files.
"""

import os
import yaml
from typing import Dict, Any, List, Optional
import argparse
from loguru import logger

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate that the configuration contains all required fields.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if the configuration is valid
    """
    required_sections = ['model_settings', 'search_settings', 'steering_settings', 'experiment_settings']
    
    for section in required_sections:
        if section not in config:
            logger.error(f"Configuration missing required section: {section}")
            return False
    
    # Check required fields in each section
    required_fields = {
        'model_settings': ['policy_model_path', 'reward_model_path'],
        'search_settings': ['beam_size', 'max_depth', 'temperature', 'num_actions'],
        'steering_settings': ['steering_strategy', 'steering_scale', 'layers_to_use', 'components_to_use'],
        'experiment_settings': ['prompt_path', 'output_path']
    }
    
    for section, fields in required_fields.items():
        for field in fields:
            if field not in config[section]:
                logger.error(f"Configuration section '{section}' missing required field: {field}")
                return False
    
    return True

def config_to_args(config: Dict[str, Any]) -> argparse.Namespace:
    """
    Convert a configuration dictionary to an argparse.Namespace object
    for compatibility with existing code.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        argparse.Namespace object containing the configuration
    """
    args = argparse.Namespace()
    
    # Flatten the configuration
    for section in config:
        for key, value in config[section].items():
            setattr(args, key.replace('-', '_'), value)
    
    return args

def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a summary of the configuration.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Configuration summary:")
    
    for section in config:
        logger.info(f"  {section}:")
        for key, value in config[section].items():
            # Format lists nicely
            if isinstance(value, list):
                value_str = ', '.join(map(str, value))
            else:
                value_str = str(value)
            
            logger.info(f"    {key}: {value_str}") 