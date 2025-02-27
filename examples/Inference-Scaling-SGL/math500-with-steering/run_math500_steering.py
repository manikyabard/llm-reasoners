#!/usr/bin/env python3

"""
Run script for math problem solving with steering-enabled beam search.
This script evaluates the effectiveness of different steering strategies
on math problem solving performance.
"""

import argparse
import json
import os
import time
import traceback
import multiprocessing
from functools import partial
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

import torch
from datasets import load_dataset
from loguru import logger
from tqdm.auto import tqdm
import nnsight
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

from reasoners import Reasoner
from reasoners.lm import HFModel, SGLangModel
from reasoners.visualization import visualize
from qwen_math_parser import math_equal

from steering_beam_search import SteeringBeamSearch
from steering_math_model import SteeringMathModel
from steering_config import SteeringMathConfig
from config_loader import load_config, validate_config, config_to_args, print_config_summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Math 500 Steering Evaluation Script')
    
    # Configuration file
    parser.add_argument('--config', type=str, 
                      help='Path to YAML configuration file')
    
    # Model paths and API endpoints
    parser.add_argument('--policy-model-path', type=str, default="/data/manikya/Llama-3.2-1B-Instruct",
                      help='Path to policy model')
    parser.add_argument('--reward-model-path', type=str, default="/data/manikya/math-shepherd-mistral-7b-prm",
                      help='Path to reward model')
    parser.add_argument('--sglang-url', type=str, default=None,
                      help='Optional SGLang API URL for remote models')
    
    # File paths
    parser.add_argument('--prompt-path', type=str,
                      help='Path to prompts JSON file')
    parser.add_argument('--output-path', type=str, default='steering_results.json',
                      help='Path to save results (default: steering_results.json)')
    parser.add_argument('--log-file', type=str, default='steering_output.log',
                      help='Path to log file (default: steering_output.log)')
    parser.add_argument('--cache-dir', type=str,
                      help='Path to cache directory (default: ./cache)')
    
    # Experiment settings
    parser.add_argument('--beam-size', type=int, default=3,
                      help='Beam size for search (default: 3)')
    parser.add_argument('--max-depth', type=int, default=10,
                      help='Maximum search depth (default: 10)')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for sampling (default: 0.7)')
    parser.add_argument('--num-actions', type=int, default=5,
                        help='Number of initial actions to consider (default: 5)')
    parser.add_argument('--num-steered-actions', type=int, default=5,
                        help='Number of steered actions to generate (default: 5)')
    parser.add_argument('--num-examples', type=int, default=5,
                        help='Number of examples to process (default: 5)')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Starting index in the dataset (default: 0)')
    parser.add_argument('--num-processes', type=int, default=4,
                        help='Number of parallel processes (default: 4)')
    
    # Steering settings
    parser.add_argument('--steering-strategy', type=str, default='reward_weighted',
                        choices=['reward_weighted', 'best_only', 'contrast', 'none'],
                        help='Strategy for computing steering vectors (default: reward_weighted)')
    parser.add_argument('--steering-scale', type=float, default=1.0,
                        help='Scaling factor for steering vectors (default: 1.0)')
    parser.add_argument('--layers-to-use', type=str, default='13,14,15',
                        help='Comma-separated list of layers to use for steering (default: -1,-2,-3)')
    parser.add_argument('--components-to-use', type=str, default='self_attn.o_proj',
                        help='Comma-separated list of component paths (default: self_attn.o_proj)')
    parser.add_argument('--num-steering-candidates', type=int, default=5,
                        help='Number of candidates to use for steering vector computation (default: 5)')
    parser.add_argument('--combine-actions', type=bool, default=True,
                        help='Whether to combine initial and steered actions (default: True)')
    parser.add_argument('--ablation-study', action='store_true',
                        help='Run ablation study comparing different steering strategies')
    
    # Hardware settings
    parser.add_argument('--device', type=str, default='cuda:0',
                      help='Device to run models on (default: cuda:0)')
    
    args = parser.parse_args()
    
    # If a configuration file is provided, load it and override command line arguments
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        if not validate_config(config):
            logger.error("Invalid configuration file. Exiting.")
            exit(1)
        
        print_config_summary(config)
        
        # Convert config to args
        config_args = config_to_args(config)
        
        # Override with config values
        for key, value in vars(config_args).items():
            if value is not None:  # Only override if value is provided in config
                setattr(args, key, value)
    
    # Verify required arguments are provided
    if not args.prompt_path:
        logger.error("No prompt path provided. Use --prompt-path or specify in config file.")
        exit(1)
    
    return args


def setup_logging(log_file):
    """Configure logging."""
    logger.remove()
    logger.add(log_file, enqueue=True)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")


def load_models(args):
    """Load language models for policy and reward."""
    # Load policy model with NNsight
    if args.sglang_url:
        # Use SGLang for remote models
        os.environ["OPENAI_API_KEY"] = "dummy"
        os.environ["SGLANG_API_URL"] = args.sglang_url
        policy_model = SGLangModel(model="", is_instruct_model=True, url=args.sglang_url)
        reward_model = SGLangModel(model="", is_instruct_model=False, url=args.sglang_url)
    else:
        # Use local models with NNsight for policy and HuggingFace for reward
        policy_model = nnsight.LanguageModel(args.policy_model_path, device_map=args.device)
        reward_model = HFModel(
            model_pth=args.reward_model_path,
            tokenizer_pth=args.reward_model_path,
            device=args.device
        )

    return policy_model, reward_model


def parse_steering_config(args):
    """Parse and prepare steering configuration from arguments."""
    # Default to no steering if strategy is 'none'
    use_steering = args.steering_strategy != 'none'
    
    # Parse layers and components
    if isinstance(args.layers_to_use, str):
        layers_to_use = [int(layer) for layer in args.layers_to_use.split(',')]
    else:
        layers_to_use = args.layers_to_use
    if isinstance(args.components_to_use, str):
        components_to_use = args.components_to_use.split(',')
    else:
        components_to_use = args.components_to_use
    
    # Create steering configuration
    steering_config = {
        "use_steering": use_steering,
        "steering_strategy": args.steering_strategy,
        "steering_scale": args.steering_scale,
        "layers_to_use": layers_to_use,
        "components_to_use": components_to_use,
        "temperature": args.temperature,
        "num_steering_candidates": min(args.num_steering_candidates, args.num_actions),
        "num_steered_generations": args.num_steered_actions,
        "combine_initial_and_steered": args.combine_actions
    }
    
    return steering_config


def setup_reasoner(policy_model, reward_model, prompt, steering_config, args):
    """Set up reasoner with steering-enabled components."""
    # Create search configuration
    config = SteeringMathConfig(
        policy_model=policy_model,
        reward_model=reward_model,
        prompt=prompt,
        steering_config=steering_config,
        num_actions=args.num_actions,
        temperature=args.temperature,
        device=args.device
    )
    
    # Create search algorithm
    search_algorithm = SteeringBeamSearch(
        beam_size=args.beam_size,
        max_depth=args.max_depth,
        steering_config=steering_config,
        sampling_strategy='stochastic' if args.temperature > 0.01 else 'argmax',
        temperature=args.temperature
    )
    
    # Create world model
    world_model = SteeringMathModel(
        policy_model=policy_model,
        reward_model=reward_model,
        prompt=prompt,
        steering_config=steering_config,
        device=args.device
    )

    # Create reasoner
    return Reasoner(
        world_model=world_model,
        search_config=config,
        search_algo=search_algorithm,
    )


def eval_answer(ground_truth, predicted):
    """Evaluate if the predicted answer matches the ground truth."""
    ground_truth = ground_truth.replace(" ", "")
    predicted = predicted.replace(" ", "")
    return math_equal(ground_truth, predicted)


def _append_to_temp_file(temp_file: str, entry: Dict):
    """Atomically append an entry to the temp JSON file"""
    temp_write = f"{temp_file}.tmp"
    
    try:
        # Read existing data
        with open(temp_file, "r") as f:
            data = json.load(f)
        
        # Ensure proper structure
        if "results" not in data:
            data["results"] = []
        
        # Append new entry
        data["results"].append(entry)
        
        # Write temporary file
        with open(temp_write, "w") as f:
            json.dump(data, f, indent=4)
        
        # Atomic replace
        os.replace(temp_write, temp_file)
    except json.JSONDecodeError:
        logger.error(f"Corrupt temp file {temp_file}, resetting")
        with open(temp_file, "w") as f:
            json.dump({"results": [entry]}, f, indent=4)
    except Exception as e:
        if os.path.exists(temp_write):
            os.remove(temp_write)
        raise


def _update_temp_file(temp_file: str, updates: Dict):
    """Update specific fields in the temp JSON file"""
    temp_write = f"{temp_file}.tmp"
    
    try:
        with open(temp_file, "r") as f:
            data = json.load(f)
        
        # Apply updates using dot notation
        for key, value in updates.items():
            keys = key.split('.')
            current = data
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = value
        
        with open(temp_write, "w") as f:
            json.dump(data, f, indent=4)
        
        os.replace(temp_write, temp_file)
    except Exception as e:
        logger.error(f"Error updating temp file {temp_file}: {e}")
        if os.path.exists(temp_write):
            os.remove(temp_write)


def process_chunk(chunk_idx: int, chunk: List[Dict[str, Any]], args: argparse.Namespace, 
                  prompt: Dict, steering_config: Optional[Dict] = None) -> Tuple[List[Dict], List[float]]:
    """Process a chunk of examples with proper JSON formatting and error handling"""
    temp_file = f"{args.output_path}.chunk_{chunk_idx}.json"
    chunk_start = time.time()
    times = []
    
    # Initialize JSON structure
    initial_data = {
        "metadata": {
            "chunk_idx": chunk_idx,
            "total_examples": len(chunk),
            "chunk_start": chunk_start,
            "steering_config": steering_config,
            "chunk_duration": None,
            "status": "processing"
        },
        "results": []
    }
    
    if not os.path.exists(temp_file):
        with open(temp_file, "w") as f:
            json.dump(initial_data, f, indent=4)

    _update_temp_file(temp_file, {"metadata.status": "running"})

    for idx, example in enumerate(chunk):
        result_entry = {
            "problem_id": example.get("id", hash(example["problem"])),
            "problem": example["problem"],
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "steering_config": steering_config,
            "status": "processing"
        }
        
        # Update the temp file with the current example being processed
        _append_to_temp_file(temp_file, result_entry)
        
        start_time = time.time()
        try:
            # Load models and setup reasoner for this example
            policy_model, reward_model = load_models(args)
            reasoner = setup_reasoner(policy_model, reward_model, prompt, steering_config, args)
            
            # Set up the problem
            problem = {"init": example["problem"]}
            reasoner.search_config.set_example(problem)
            reasoner.dynamics.set_question(problem)
            
            # Run the reasoner
            trace = reasoner(problem)
            solution = trace.terminal_state.solution
            steps = trace.terminal_state.steps
            
            # Extract metrics
            rewards = [action[2] for action in trace.trace if action[2] is not None]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            
            # Get steering metrics if available
            steering_metrics = getattr(trace, "steering_metrics", None)
            
            # Update result with success information
            result_entry.update({
                "status": "completed",
                "predicted_answer": solution,
                "predicted_steps": steps,
                "ground_truth": {
                    "steps": example["solution"].split("\n"),
                    "answer": example["answer"]
                },
                "is_correct": eval_answer(solution, example["answer"]),
                "processing_time": time.time() - start_time,
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "metrics": {
                    "num_steps": len(steps),
                    "rewards": rewards,
                    "avg_reward": avg_reward
                }
            })
            
            # Add steering metrics to the result if available
            if steering_metrics:
                result_entry["steering_metrics"] = steering_metrics
            
        except Exception as e:
            # Update result with error information
            result_entry.update({
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "processing_time": time.time() - start_time,
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            })
            logger.error(f"Error processing example {result_entry['problem_id']} in chunk {chunk_idx}: {str(e)}")
        
        # Update the entry in the temp file
        _update_temp_file(temp_file, {
            f"results[{idx}]": result_entry
        })
        
        # Track processing time
        times.append(result_entry.get("processing_time", 0))
        
    # Final chunk metadata update
    _update_temp_file(temp_file, {
        "metadata.status": "completed",
        "metadata.chunk_duration": time.time() - chunk_start
    })

    return [r for r in initial_data.get("results", []) if r.get("status") == "completed"], times


def process_ablation_chunk(chunk_idx: int, chunk: List[Dict[str, Any]], args: argparse.Namespace, 
                          prompt: Dict, steering_configs: List[Dict]) -> Tuple[List[Dict], List[float]]:
    """Process a chunk of examples with all steering configurations for ablation study"""
    temp_file = f"{args.output_path}.ablation_chunk_{chunk_idx}.json"
    chunk_start = time.time()
    times = []
    
    # Initialize JSON structure
    initial_data = {
        "metadata": {
            "chunk_idx": chunk_idx,
            "total_examples": len(chunk),
            "chunk_start": chunk_start,
            "steering_configs": steering_configs,
            "chunk_duration": None,
            "status": "processing"
        },
        "results": []
    }
    
    if not os.path.exists(temp_file):
        with open(temp_file, "w") as f:
            json.dump(initial_data, f, indent=4)

    _update_temp_file(temp_file, {"metadata.status": "running"})
    
    for example_idx, example in enumerate(chunk):
        example_results = []
        
        for config_idx, config_info in enumerate(steering_configs):
            result_entry = {
                "problem_id": example.get("id", hash(example["problem"])),
                "problem": example["problem"],
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "steering_strategy": config_info["name"],
                "steering_config": config_info["config"],
                "status": "processing"
            }
            
            # Update the temp file with the current example being processed
            _append_to_temp_file(temp_file, result_entry)
            
            start_time = time.time()
            try:
                # Load models and setup reasoner for this example
                policy_model, reward_model = load_models(args)
                reasoner = setup_reasoner(policy_model, reward_model, prompt, config_info["config"], args)
                
                # Set up the problem
                problem = {"init": example["problem"]}
                reasoner.search_config.set_example(problem)
                reasoner.dynamics.set_question(problem)
                
                # Run the reasoner
                trace = reasoner(problem)
                solution = trace.terminal_state.solution
                steps = trace.terminal_state.steps
                
                # Extract metrics
                rewards = [action[2] for action in trace.trace if action[2] is not None]
                avg_reward = sum(rewards) / len(rewards) if rewards else 0
                
                # Get steering metrics if available
                steering_metrics = getattr(trace, "steering_metrics", None)
                
                # Update result with success information
                result_entry.update({
                    "status": "completed",
                    "predicted_answer": solution,
                    "predicted_steps": steps,
                    "ground_truth": {
                        "steps": example["solution"].split("\n"),
                        "answer": example["answer"]
                    },
                    "is_correct": eval_answer(solution, example["answer"]),
                    "processing_time": time.time() - start_time,
                    "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "metrics": {
                        "num_steps": len(steps),
                        "rewards": rewards,
                        "avg_reward": avg_reward
                    }
                })
                
                # Add steering metrics to the result if available
                if steering_metrics:
                    result_entry["steering_metrics"] = steering_metrics
                
            except Exception as e:
                # Update result with error information
                result_entry.update({
                    "status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "processing_time": time.time() - start_time,
                    "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                })
                logger.error(f"Error processing example {result_entry['problem_id']} with strategy {config_info['name']}: {str(e)}")
            
            # Update the entry in the temp file
            result_idx = len(initial_data.get("results", [])) - 1
            _update_temp_file(temp_file, {
                f"results[{result_idx}]": result_entry
            })
            
            # Track processing time
            times.append(result_entry.get("processing_time", 0))
            example_results.append(result_entry)
        
        # Log comparison for this example
        if any(r["status"] == "completed" for r in example_results):
            logger.info(f"\nResults comparison for example {example.get('id', hash(example['problem']))}:")
            
            for result in example_results:
                if result["status"] == "completed":
                    logger.info(f"  {result['steering_strategy']}:")
                    logger.info(f"    Correct: {result['is_correct']}")
                    logger.info(f"    Avg Reward: {result['metrics']['avg_reward']:.4f}")
                    logger.info(f"    Num Steps: {result['metrics']['num_steps']}")
    
    # Final chunk metadata update
    _update_temp_file(temp_file, {
        "metadata.status": "completed",
        "metadata.chunk_duration": time.time() - chunk_start
    })

    return [r for r in initial_data.get("results", []) if r.get("status") == "completed"], times


def aggregate_steering_metrics(results):
    """Aggregate steering metrics across multiple examples."""
    aggregated = {
        "total_examples": 0,
        "examples_with_steering": 0,
        "total_steps": 0,
        "steps_with_steering": 0,
        "initial_actions_selected": 0,
        "steered_actions_selected": 0,
        "steps_with_steered_improvement": 0,
        "avg_initial_reward": [],
        "avg_steered_reward": [],
        "mean_reward_improvement": []
    }
    
    for result in results:
        if result.get("status") == "completed" and "steering_metrics" in result:
            metrics = result["steering_metrics"]
            aggregated["total_examples"] += 1
            
            if metrics.get("steps_with_steering", 0) > 0:
                aggregated["examples_with_steering"] += 1
            
            # Aggregate step counts
            aggregated["total_steps"] += metrics.get("total_steps", 0)
            aggregated["steps_with_steering"] += metrics.get("steps_with_steering", 0)
            
            # Aggregate action selection counts
            aggregated["initial_actions_selected"] += metrics.get("initial_actions_selected", 0)
            aggregated["steered_actions_selected"] += metrics.get("steered_actions_selected", 0)
            
            # Aggregate improvement counts
            aggregated["steps_with_steered_improvement"] += metrics.get("steps_with_steered_improvement", 0)
            
            # Track reward metrics
            if "avg_initial_reward" in metrics:
                aggregated["avg_initial_reward"].append(metrics["avg_initial_reward"])
            if "avg_steered_reward" in metrics:
                aggregated["avg_steered_reward"].append(metrics["avg_steered_reward"])
            if "mean_reward_improvement" in metrics:
                aggregated["mean_reward_improvement"].append(metrics["mean_reward_improvement"])
    
    # Calculate final aggregated statistics
    if aggregated["total_examples"] > 0:
        # Calculate rates
        aggregated["examples_with_steering_rate"] = aggregated["examples_with_steering"] / aggregated["total_examples"]
        
        if aggregated["total_steps"] > 0:
            aggregated["steps_with_steering_rate"] = aggregated["steps_with_steering"] / aggregated["total_steps"]
        else:
            aggregated["steps_with_steering_rate"] = 0.0
            
        total_actions = aggregated["initial_actions_selected"] + aggregated["steered_actions_selected"]
        if total_actions > 0:
            aggregated["steered_selection_rate"] = aggregated["steered_actions_selected"] / total_actions
        else:
            aggregated["steered_selection_rate"] = 0.0
            
        if aggregated["steps_with_steering"] > 0:
            aggregated["steered_improvement_rate"] = aggregated["steps_with_steered_improvement"] / aggregated["steps_with_steering"]
        else:
            aggregated["steered_improvement_rate"] = 0.0
            
        # Calculate average rewards
        if aggregated["avg_initial_reward"]:
            aggregated["avg_initial_reward_value"] = np.mean(aggregated["avg_initial_reward"])
            del aggregated["avg_initial_reward"]
        else:
            aggregated["avg_initial_reward_value"] = 0.0
            
        if aggregated["avg_steered_reward"]:
            aggregated["avg_steered_reward_value"] = np.mean(aggregated["avg_steered_reward"])
            del aggregated["avg_steered_reward"]
        else:
            aggregated["avg_steered_reward_value"] = 0.0
            
        if aggregated["mean_reward_improvement"]:
            aggregated["mean_reward_improvement_value"] = np.mean(aggregated["mean_reward_improvement"])
            del aggregated["mean_reward_improvement"]
        else:
            aggregated["mean_reward_improvement_value"] = 0.0
    
    return aggregated


def log_steering_effectiveness(metrics):
    """Log detailed metrics about steering effectiveness."""
    logger.info("\n=== Steering Effectiveness Metrics ===")
    
    # Usage statistics
    logger.info(f"Steps using steering: {metrics['steps_with_steering']} / {metrics['total_steps']} ({metrics['steering_usage_rate']:.1%})")
    
    # Selection statistics
    total_selected = metrics['initial_actions_selected'] + metrics['steered_actions_selected']
    logger.info(f"Action selection: Steered: {metrics['steered_actions_selected']} / {total_selected} ({metrics['steered_selection_rate']:.1%})")
    
    # Reward improvement statistics
    logger.info(f"Steps with steered improvement: {metrics['steps_with_steered_improvement']} / {metrics['steps_with_steering']} ({metrics['steered_improvement_rate']:.1%})")
    logger.info(f"Average reward: Initial: {metrics['avg_initial_reward']:.4f}, Steered: {metrics['avg_steered_reward']:.4f}")
    logger.info(f"Mean reward improvement: {metrics['mean_reward_improvement']:.4f}")
    
    # Best action source
    logger.info(f"Steered actions were best: {metrics['steered_best_rate']:.1%} of the time")
    
    logger.info("========================================\n")


def log_aggregated_steering_metrics(aggregated_metrics):
    """Log aggregated steering metrics across all examples."""
    logger.info("\n=== Aggregated Steering Effectiveness Metrics ===")
    
    # Example and step statistics
    logger.info(f"Examples using steering: {aggregated_metrics['examples_with_steering']} / {aggregated_metrics['total_examples']} ({aggregated_metrics['examples_with_steering_rate']:.1%})")
    logger.info(f"Steps using steering: {aggregated_metrics['steps_with_steering']} / {aggregated_metrics['total_steps']} ({aggregated_metrics['steps_with_steering_rate']:.1%})")
    
    # Selection statistics
    total_actions = aggregated_metrics['initial_actions_selected'] + aggregated_metrics['steered_actions_selected']
    logger.info(f"Action selection: Steered: {aggregated_metrics['steered_actions_selected']} / {total_actions} ({aggregated_metrics['steered_selection_rate']:.1%})")
    
    # Reward improvement statistics
    logger.info(f"Steps with steered improvement: {aggregated_metrics['steps_with_steered_improvement']} / {aggregated_metrics['steps_with_steering']} ({aggregated_metrics['steered_improvement_rate']:.1%})")
    logger.info(f"Average reward: Initial: {aggregated_metrics['avg_initial_reward_value']:.4f}, Steered: {aggregated_metrics['avg_steered_reward_value']:.4f}")
    logger.info(f"Mean reward improvement: {aggregated_metrics['mean_reward_improvement_value']:.4f}")
    
    logger.info("=================================================\n")


def main():
    """Main entry point with parallelization."""
    total_start = time.time()
    args = parse_args()
    setup_logging(args.log_file)
    
    logger.info(f"Starting math500 steering evaluation with {args.num_examples} examples")
    logger.info(f"Beam size: {args.beam_size}, Max depth: {args.max_depth}")
    logger.info(f"Using policy model: {args.policy_model_path}")
    logger.info(f"Using reward model: {args.reward_model_path}")
    
    if hasattr(args, 'config') and args.config:
        logger.info(f"Using configuration file: {args.config}")
        
    # Load prompts
    with open(args.prompt_path, "r") as f:
        prompt = json.load(f)
    
    # Load dataset
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=args.cache_dir)
    dataset_list = [ex for ex in dataset]
    
    # Apply filters and limits
    start_idx = args.start_idx
    end_idx = min(start_idx + args.num_examples, len(dataset_list)) if args.num_examples else len(dataset_list)
    examples = dataset_list[start_idx:end_idx]
    
    logger.info(f"Loaded {len(examples)} examples from dataset (index {start_idx} to {end_idx-1})")
    
    # Create processing chunks
    num_processes = min(args.num_processes, len(examples))  # Ensure we don't create more processes than examples
    chunk_size = max(1, len(examples) // num_processes)     # Ensure chunk size is at least 1
    chunks = [examples[i*chunk_size : (i+1)*chunk_size] for i in range(num_processes)]
    
    # Distribute remainder examples
    remainder = len(examples) % num_processes
    for i in range(remainder):
        if i < len(chunks):
            chunks[i].append(examples[num_processes * chunk_size + i])
    
    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk]
    chunks_with_indices = list(enumerate(chunks))
    
    if args.ablation_study:
        # Define steering configurations to compare
        steering_configs = [
            # No steering (baseline)
            {
                "name": "No Steering",
                "config": {
                    "use_steering": False,
                    "steering_strategy": "none",
                    "layers_to_use": [],
                    "components_to_use": []
                }
            },
            # Reward-weighted strategy
            {
                "name": "Reward Weighted",
                "config": {
                    "use_steering": True,
                    "steering_strategy": "reward_weighted",
                    "steering_scale": args.steering_scale,
                    "layers_to_use": [int(layer) for layer in args.layers_to_use.split(',')] if isinstance(args.layers_to_use, str) else args.layers_to_use,
                    "components_to_use": args.components_to_use.split(',') if isinstance(args.components_to_use, str) else args.components_to_use,
                    "temperature": args.temperature
                }
            },
            # Best-only strategy
            {
                "name": "Best Only",
                "config": {
                    "use_steering": True,
                    "steering_strategy": "best_only",
                    "steering_scale": args.steering_scale,
                    "layers_to_use": [int(layer) for layer in args.layers_to_use.split(',')] if isinstance(args.layers_to_use, str) else args.layers_to_use,
                    "components_to_use": args.components_to_use.split(',') if isinstance(args.components_to_use, str) else args.components_to_use,
                    "temperature": args.temperature
                }
            },
            # Contrast strategy
            {
                "name": "Contrast",
                "config": {
                    "use_steering": True,
                    "steering_strategy": "contrast",
                    "steering_scale": args.steering_scale,
                    "layers_to_use": [int(layer) for layer in args.layers_to_use.split(',')] if isinstance(args.layers_to_use, str) else args.layers_to_use,
                    "components_to_use": args.components_to_use.split(',') if isinstance(args.components_to_use, str) else args.components_to_use,
                    "temperature": args.temperature
                }
            }
        ]
        
        logger.info("Running ablation study with different steering strategies")
        process_func = partial(process_ablation_chunk, args=args, prompt=prompt, steering_configs=steering_configs)
        
        try:
            with multiprocessing.Pool(processes=args.num_processes) as pool:
                results = list(tqdm(
                    pool.starmap(process_func, chunks_with_indices),
                    total=len(chunks_with_indices),
                    desc="Processing ablation chunks"
                ))
        except Exception as e:
            logger.error(f"Error in ablation study multiprocessing: {e}")
            traceback.print_exc()
        
        # Aggregate results from all chunks
        all_results = []
        
        # Load and merge chunk files
        chunk_files = [f for f in os.listdir() if f.startswith(f"{args.output_path}.ablation_chunk_")]
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, "r") as f:
                    chunk_data = json.load(f)
                    all_results.extend(chunk_data.get("results", []))
                os.remove(chunk_file)
            except Exception as e:
                logger.error(f"Error processing {chunk_file}: {str(e)}")
        
        # Save results
        with open(args.output_path, "w") as f:
            json.dump({
                "metadata": {
                    "args": vars(args),
                    "steering_configs": steering_configs,
                    "total_time": time.time() - total_start
                },
                "results": all_results
            }, f, indent=2)
        
        # Aggregate results by strategy for reporting
        strategy_stats = {}
        for config_info in steering_configs:
            strategy_name = config_info["name"]
            strategy_results = [r for r in all_results if r.get("steering_strategy") == strategy_name]
            
            completed = [r for r in strategy_results if r["status"] == "completed"]
            correct = [r for r in completed if r["is_correct"]]
            
            if completed:
                avg_reward = sum(r["metrics"]["avg_reward"] for r in completed) / len(completed)
                avg_steps = sum(r["metrics"]["num_steps"] for r in completed) / len(completed)
                avg_time = sum(r["processing_time"] for r in completed) / len(completed)
                
                # Aggregate steering metrics for this strategy
                strategy_steering_metrics = None
                if strategy_name != "No Steering":
                    strategy_steering_results = [r for r in completed if "steering_metrics" in r]
                    if strategy_steering_results:
                        strategy_steering_metrics = aggregate_steering_metrics(strategy_steering_results)
            else:
                avg_reward = 0.0
                avg_steps = 0.0
                avg_time = 0.0
                strategy_steering_metrics = None
            
            strategy_stats[strategy_name] = {
                "total": len(strategy_results),
                "completed": len(completed),
                "correct": len(correct),
                "success_rate": len(correct) / len(completed) if completed else 0.0,
                "avg_reward": avg_reward,
                "avg_steps": avg_steps,
                "avg_time": avg_time,
                "steering_metrics": strategy_steering_metrics
            }
        
        # Print summary
        logger.info("\n============== ABLATION STUDY SUMMARY ==============")
        for strategy, stats in strategy_stats.items():
            logger.info(f"\n{strategy}:")
            logger.info(f"  Success Rate: {stats['success_rate']:.2%} ({stats['correct']}/{stats['completed']})")
            logger.info(f"  Avg Reward: {stats['avg_reward']:.4f}")
            logger.info(f"  Avg Steps: {stats['avg_steps']:.2f}")
            logger.info(f"  Avg Time: {stats['avg_time']:.2f}s")
            
            # Log steering metrics summary if available
            if stats['steering_metrics']:
                logger.info(f"  Steering Effectiveness:")
                logger.info(f"    Steered selection rate: {stats['steering_metrics']['steered_selection_rate']:.1%}")
                logger.info(f"    Steps with improvement: {stats['steering_metrics']['steered_improvement_rate']:.1%}")
                logger.info(f"    Mean reward improvement: {stats['steering_metrics']['mean_reward_improvement_value']:.4f}")
    
    else:
        # Process examples with the specified steering configuration
        steering_config = parse_steering_config(args)
        logger.info(f"Processing examples with steering strategy: {args.steering_strategy}")
        
        process_func = partial(process_chunk, args=args, prompt=prompt, steering_config=steering_config)
        
        try:
            with multiprocessing.Pool(processes=args.num_processes) as pool:
                results = list(tqdm(
                    pool.starmap(process_func, chunks_with_indices),
                    total=len(chunks_with_indices),
                    desc="Processing chunks"
                ))
        except Exception as e:
            logger.error(f"Error in multiprocessing: {e}")
            traceback.print_exc()
        
        # Aggregate results from all chunks
        all_results = []
        
        # Load and merge chunk files
        chunk_files = [f for f in os.listdir() if f.startswith(f"{args.output_path}.chunk_")]
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, "r") as f:
                    chunk_data = json.load(f)
                    all_results.extend(chunk_data.get("results", []))
                os.remove(chunk_file)
            except Exception as e:
                logger.error(f"Error processing {chunk_file}: {str(e)}")
        
        # Save results
        with open(args.output_path, "w") as f:
            json.dump({
                "metadata": {
                    "args": vars(args),
                    "steering_config": steering_config,
                    "total_time": time.time() - total_start
                },
                "results": all_results
            }, f, indent=2)
        
        # Calculate and print summary statistics
        completed = [r for r in all_results if r["status"] == "completed"]
        correct = [r for r in completed if r["is_correct"]]
        
        if completed:
            avg_reward = sum(r["metrics"]["avg_reward"] for r in completed) / len(completed)
            avg_steps = sum(r["metrics"]["num_steps"] for r in completed) / len(completed)
            avg_time = sum(r["processing_time"] for r in completed) / len(completed)
            
            # Aggregate steering metrics if available
            if args.steering_strategy != "none":
                completed_with_steering = [r for r in completed if "steering_metrics" in r]
                if completed_with_steering:
                    aggregated_steering_metrics = aggregate_steering_metrics(completed_with_steering)
                    log_steering_effectiveness(aggregated_steering_metrics)
                    log_aggregated_steering_metrics(aggregated_steering_metrics)
        else:
            avg_reward = 0.0
            avg_steps = 0.0
            avg_time = 0.0
        
        logger.info("\n============== SUMMARY ==============")
        logger.info(f"Total Examples: {len(all_results)}")
        logger.info(f"Completed: {len(completed)}")
        logger.info(f"Correct: {len(correct)}")
        logger.info(f"Success Rate: {len(correct) / len(completed) if completed else 0.0:.2%}")
        logger.info(f"Avg Reward: {avg_reward:.4f}")
        logger.info(f"Avg Steps: {avg_steps:.2f}")
        logger.info(f"Avg Time: {avg_time:.2f}s")
        logger.info(f"Total Time: {time.time() - total_start:.2f}s")
    
    logger.info("Done!")


if __name__ == "__main__":
    total_start = time.time()
    main() 