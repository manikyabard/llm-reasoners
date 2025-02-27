"""
Test script for enhanced steering approach.
This script tests different layer combinations for steering.
"""

import torch
import numpy as np
from steering_poc import SteeringPOC
from loguru import logger
from datasets import load_dataset
import argparse
from transformers import AutoModelForCausalLM
import nnsight
import sys
import logging


def test_layer_combinations(problem, target_step=1, device="cuda:2", debug=False):
    """Test different layer combinations for steering.
    
    Args:
        problem: The math problem to solve
        target_step: The specific step to steer (default: 1)
        device: Device to use for computation
    """
    logger.info(f"Testing different layer combinations for steering step {target_step}...")
    
    # Define layer combinations to test
    layer_combinations = [
        {"name": "Late layers only", "layers": [-1, -2, -3]},
        {"name": "Early layers only", "layers": [0, 1, 2]},
        {"name": "Middle layers only", "layers": [4, 5, 6, 7]},
        {"name": "Mixed layers", "layers": [0, 3, 6, 9, -3, -1]},
        {"name": "All layers (sparse)", "layers": [0, 3, 6, 9, 12, 15, -3, -1]}
    ]
    
    # Define component combinations to test
    component_combinations = [
        {"name": "Attention output only", "components": ["self_attn.o_proj"]},
        {"name": "Full attention", "components": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]},
        {"name": "MLP components", "components": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]},
        {"name": "Output projections", "components": ["self_attn.o_proj", "mlp.down_proj"]}
    ]
    
    results = {}
    
    # Collect all layers and components to capture in one go
    all_layers = []
    all_components = []
    for layer_config in layer_combinations:
        all_layers.extend(layer_config["layers"])
    for component_config in component_combinations:
        all_components.extend(component_config["components"])
    
    # Remove duplicates while preserving order
    all_layers = list(dict.fromkeys(all_layers))
    all_components = list(dict.fromkeys(all_components))
    
    logger.info(f"Capturing activations for all layers: {all_layers}")
    logger.info(f"Capturing activations for all components: {all_components}")
    
    poc_object = None
    # Initialize base steering object with all layers and components
    policy_model = nnsight.LanguageModel("/data/manikya/Llama-3.2-1B-Instruct", device_map=device)
    reward_model = AutoModelForCausalLM.from_pretrained("/data/manikya/math-shepherd-mistral-7b-prm").eval().to(device)

    base_steering = SteeringPOC(
        policy_model_path="/data/manikya/Llama-3.2-1B-Instruct",
        reward_model_path="/data/manikya/math-shepherd-mistral-7b-prm",
        policy_model=policy_model,
        reward_model=reward_model,
        device=device,
        num_generations=5,
        temperature=0.7,
        layers_to_use=all_layers,
        components_to_use=all_components
    )
    poc_object = base_steering
    
    # Run the initial POC to capture all activations for the target step
    logger.info(f"Generating original outputs and capturing activations for step {target_step}...")
    base_result = base_steering.run_poc(problem, target_step=target_step)
    
    # Extract the data we need for reuse
    original_generations = base_result["original_generations"]
    original_rewards = base_result["original_rewards"]
    activations_list = base_result["activations_list"]
    
    # Check if we have enough valid generations for the target step
    if len(original_rewards) < 2:
        logger.warning(f"Not enough valid generations with target step {target_step}. Try a different step or problem.")
        return {}, {}
    
    # Convert tensor rewards to CPU numpy if needed
    original_rewards = [r.cpu().item() if torch.is_tensor(r) else r for r in original_rewards]
    avg_original_reward = np.mean(original_rewards)
    
    logger.info(f"Original generations complete. Average reward for step {target_step}: {avg_original_reward:.4f}")
    
    # Test each layer combination with default components
    logger.info("Testing layer combinations with default component (attention output)...")
    for layer_config in layer_combinations:
        logger.info(f"Testing {layer_config['name']}: {layer_config['layers']}")
        
        # Initialize steering with this layer combination
        poc_object = SteeringPOC(
            policy_model_path="/data/manikya/Llama-3.2-1B-Instruct",
            reward_model_path="/data/manikya/math-shepherd-mistral-7b-prm",
            policy_model=policy_model,
            reward_model=reward_model,
            device=device,
            num_generations=5,
            temperature=0.7,
            layers_to_use=layer_config["layers"],
            components_to_use=["self_attn.o_proj"],  # Default to attention output only
            steering_scale=1.0,
            debug=debug
        )
        steering = poc_object
        
        # Set the activations directly from the base steering object - not needed anymore
        # Compute steering vector and generate steered output
        steering_vectors = steering.compute_steering_vector(
            activations_list,  # Use the activations list directly from run_poc
            original_rewards
        )
        
        # Debug steering vectors
        logger.debug(f"Steering vector keys for {layer_config['name']}: {list(steering_vectors.keys())}")
        for key in list(steering_vectors.keys())[:2]:  # Show first two for brevity
            logger.debug(f"Shape of {key}: {steering_vectors[key].shape}")
        
        # Generate steered completion specifically for the target step
        steered_completion, _ = steering.generate(problem, target_step=target_step, steering_vector=steering_vectors)
        
        # Log the steered completion for debugging
        logger.debug(f"Steered completion for {layer_config['name']}: {steered_completion}")
        
        # Get reward for steered completion's target step
        steered_steps = steering.extract_steps(steered_completion)
        logger.debug(f"Extracted steps: {steered_steps}")
        
        if steered_steps and len(steered_steps) >= target_step:
            # Only evaluate the target step
            steered_reward = steering.get_reward(problem, steered_steps, target_step)
            logger.debug(f"Reward for step {target_step}: {steered_reward}")
        else:
            logger.warning(f"Target step {target_step} not found in steered completion for {layer_config['name']}")
            steered_reward = 0.0
        
        # Convert tensor reward to CPU numpy if needed
        steered_reward = steered_reward.cpu().item() if torch.is_tensor(steered_reward) else steered_reward
        
        # Store results
        results[layer_config["name"]] = {
            "original_rewards": original_rewards,
            "steered_reward": steered_reward,
            "avg_original_reward": avg_original_reward,
            "improvement": steered_reward - avg_original_reward
        }
        
        logger.info(f"Results for {layer_config['name']} (step {target_step}):")
        logger.info(f"  Avg original reward: {avg_original_reward:.4f}")
        logger.info(f"  Steered reward: {steered_reward:.4f}")
        logger.info(f"  Improvement: {steered_reward - avg_original_reward:.4f}")
    
    # Test the best layer combination with different components
    best_layer_config = max(results.items(), key=lambda x: x[1]["improvement"])
    logger.info(f"Best layer configuration: {best_layer_config[0]} with improvement {best_layer_config[1]['improvement']:.4f}")
    
    # Test each component combination with the best layer config
    logger.info(f"Testing component combinations with {best_layer_config[0]}...")
    best_layers = next(lc["layers"] for lc in layer_combinations if lc["name"] == best_layer_config[0])
    
    component_results = {}
    for component_config in component_combinations:
        logger.info(f"Testing {component_config['name']}: {component_config['components']}")
        
        # Initialize steering with this component combination
        poc_object = SteeringPOC(
            policy_model_path="/data/manikya/Llama-3.2-1B-Instruct",
            reward_model_path="/data/manikya/math-shepherd-mistral-7b-prm",
            policy_model=policy_model,
            reward_model=reward_model,
            device=device,
            num_generations=5,
            temperature=0.7,
            layers_to_use=best_layers,
            components_to_use=component_config["components"],
            steering_scale=1.0,
            debug=debug
        )
        steering = poc_object
        
        # Debug the components being used
        logger.debug(f"Components being used for {component_config['name']}: {steering.components_to_use}")
        
        # Set the activations directly from the base steering object - not needed anymore
        # Compute steering vector and generate steered output
        steering_vectors = steering.compute_steering_vector(
            activations_list,  # Use the activations list directly from run_poc
            original_rewards
        )
        
        # Debug steering vectors
        logger.debug(f"Steering vector keys for {component_config['name']}: {list(steering_vectors.keys())}")
        for key in list(steering_vectors.keys())[:2]:  # Show first two for brevity
            logger.debug(f"Shape of {key}: {steering_vectors[key].shape}")
        
        # Generate steered completion specifically for the target step
        steered_completion, _ = steering.generate(problem, target_step=target_step, steering_vector=steering_vectors)
        
        # Log the steered completion for debugging
        logger.debug(f"Steered completion for {component_config['name']}: {steered_completion}")
        
        # Get reward for steered completion's target step
        steered_steps = steering.extract_steps(steered_completion)
        logger.debug(f"Extracted steps: {steered_steps}")
        
        if steered_steps and len(steered_steps) >= target_step:
            # Only evaluate the target step
            steered_reward = steering.get_reward(problem, steered_steps, target_step)
            logger.debug(f"Reward for step {target_step}: {steered_reward}")
        else:
            logger.warning(f"Target step {target_step} not found in steered completion for {component_config['name']}")
            steered_reward = 0.0
        
        # Convert tensor reward to CPU numpy if needed
        steered_reward = steered_reward.cpu().item() if torch.is_tensor(steered_reward) else steered_reward
        
        # Store results
        component_results[component_config["name"]] = {
            "original_rewards": original_rewards,
            "steered_reward": steered_reward,
            "avg_original_reward": avg_original_reward,
            "improvement": steered_reward - avg_original_reward
        }
        
        logger.info(f"Results for {component_config['name']} (step {target_step}):")
        logger.info(f"  Avg original reward: {avg_original_reward:.4f}")
        logger.info(f"  Steered reward: {steered_reward:.4f}")
        logger.info(f"  Improvement: {steered_reward - avg_original_reward:.4f}")
    
    # Output final summary
    logger.info("\nFinal Summary of Results:")
    logger.info("========================")
    logger.info(f"Layer Combination Results (step {target_step}):")
    for name, result in results.items():
        logger.info(f"  {name}: Improvement: {result['improvement']:.4f}, Steered reward: {result['steered_reward']:.4f}")
    
    logger.info(f"\nComponent Combination Results (with best layer configuration) (step {target_step}):")
    for name, result in component_results.items():
        logger.info(f"  {name}: Improvement: {result['improvement']:.4f}, Steered reward: {result['steered_reward']:.4f}")
    
    return results, component_results

def main():
    """Main function to run the POC tests."""
    parser = argparse.ArgumentParser(description='Test steering approach')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on')
    parser.add_argument('--problem_index', type=int, default=0, help='Index of problem to solve')
    parser.add_argument('--target_step', type=int, default=2, help='Target step to improve')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # global logger
    # If debug is enabled, reset logger to debug level
    # if args.debug:
    #     # # Remove existing handlers
    #     # for handler in logger.handlers[:]:
    #     #     logger.removeHandler(handler)
        
    #     # Add new handler with DEBUG level
    #     handler = logging.StreamHandler(sys.stderr)
    #     handler.setLevel(logging.DEBUG)
    #     # formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s')
    #     # handler.setFormatter(formatter)
    #     logger.addHandler(handler)
    #     logger.setLevel(logging.DEBUG)
    
    # Load a problem from the MATH-500 dataset
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir="/data/manikya/huggingface")
    problem = dataset[args.problem_index]["problem"]
    logger.info(f"Testing with problem: {problem}")
    logger.info(f"Target step: {args.target_step}")
    
    # Run tests
    layer_results, component_results = test_layer_combinations(problem, target_step=args.target_step, device=args.device, debug=args.debug)
    
    # Output best configuration
    best_layer = max(layer_results.items(), key=lambda x: x[1]["improvement"])
    best_component = max(component_results.items(), key=lambda x: x[1]["improvement"])
    
    logger.info("\n=== Best Configuration ===")
    logger.info(f"Best layer combination for step {args.target_step}: {best_layer[0]} with improvement {best_layer[1]['improvement']:.4f}")
    logger.info(f"Best component combination for step {args.target_step}: {best_component[0]} with improvement {best_component[1]['improvement']:.4f}")

if __name__ == "__main__":
    main() 