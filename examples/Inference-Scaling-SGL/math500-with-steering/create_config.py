#!/usr/bin/env python3

"""
Utility script to create new configuration files for math500-with-steering experiments.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

DEFAULT_CONFIG = {
    "model_settings": {
        "policy_model_path": "/data/manikya/Llama-3.2-1B-Instruct",
        "reward_model_path": "/data/manikya/math-shepherd-mistral-7b-prm",
        "sglang_url": None,
        "device": "cuda:0"
    },
    "search_settings": {
        "beam_size": 3,
        "max_depth": 10,
        "temperature": 0.7,
        "num_actions": 5,
        "num_steered_actions": 5
    },
    "steering_settings": {
        "steering_strategy": "reward_weighted",
        "steering_scale": 1.0,
        "layers_to_use": [-1, -2, -3],
        "components_to_use": ["self_attn.o_proj"],
        "num_steering_candidates": 5,
        "combine_actions": True
    },
    "experiment_settings": {
        "prompt_path": "path/to/prompts.json",
        "output_path": "results/results.json",
        "log_file": "logs/run.log",
        "num_examples": 5,
        "start_idx": 0,
        "num_processes": 1
    }
}

def create_config(name, base=None, description=None):
    """Create a new configuration file."""
    config_dir = Path("configs")
    os.makedirs(config_dir, exist_ok=True)
    
    config_path = config_dir / f"{name}.yaml"
    
    # Check if file already exists
    if config_path.exists():
        response = input(f"Configuration '{name}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # If base is provided, update with base configuration
    if base:
        base_path = config_dir / f"{base}.yaml"
        if not base_path.exists():
            print(f"Base configuration '{base}' not found!")
            return
        
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Update the default config with the base config
        for section in base_config:
            if section in config:
                config[section].update(base_config[section])
            else:
                config[section] = base_config[section]
    
    # Update output paths with the configuration name
    config["experiment_settings"]["output_path"] = f"results/{name}_results.json"
    config["experiment_settings"]["log_file"] = f"logs/{name}_run.log"
    
    # Write configuration file
    with open(config_path, 'w') as f:
        # Add description comment if provided
        if description:
            f.write(f"# {name} configuration for math500-with-steering experiments\n")
            f.write(f"# {description}\n\n")
        
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Configuration '{name}' created at {config_path}")
    print(f"Edit the file to customize your experiment settings.")

def main():
    parser = argparse.ArgumentParser(description="Create a new configuration file for math500-with-steering experiments")
    parser.add_argument("name", help="Name of the configuration")
    parser.add_argument("--base", help="Base configuration to extend")
    parser.add_argument("--description", help="Description of the configuration")
    
    args = parser.parse_args()
    
    create_config(args.name, args.base, args.description)

if __name__ == "__main__":
    main() 