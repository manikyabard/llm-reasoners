#!/usr/bin/env python3

"""
Utility script to compare the results of multiple math500-with-steering experiments.
This script takes multiple result JSON files and generates a comparison report.
"""

import argparse
import json
import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

def load_results(result_paths: List[str]) -> Dict[str, Any]:
    """
    Load results from multiple JSON files.
    
    Args:
        result_paths: List of paths to result JSON files
        
    Returns:
        Dictionary mapping experiment names to results
    """
    results = {}
    
    for path in result_paths:
        # Get experiment name from file path
        experiment_name = os.path.basename(path).replace("_results.json", "")
        
        try:
            with open(path, 'r') as f:
                results[experiment_name] = json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
    
    return results

def extract_metrics(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Extract key metrics from results.
    
    Args:
        results: Dictionary mapping experiment names to results
        
    Returns:
        Dictionary mapping experiment names to metrics
    """
    metrics = {}
    
    for experiment_name, result in results.items():
        # Extract success metrics
        success_count = sum(1 for r in result["results"] if r.get("is_correct", False))
        total_count = len(result["results"])
        success_rate = success_count / total_count if total_count > 0 else 0
        
        # Extract average rewards
        rewards = []
        for r in result["results"]:
            if "metrics" in r and "rewards" in r["metrics"]:
                rewards.extend(r["metrics"]["rewards"])
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        
        # Extract average processing time
        processing_times = [r.get("processing_time", 0) for r in result["results"]]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Extract steering metrics if available
        steering_metrics = {}
        if "aggregated_steering_metrics" in result:
            agg_metrics = result["aggregated_steering_metrics"]
            steering_metrics = {
                "steering_usage_rate": agg_metrics.get("steering_usage_rate", 0),
                "steered_selection_rate": agg_metrics.get("steered_selection_rate", 0),
                "mean_reward_improvement": agg_metrics.get("mean_reward_improvement", 0),
                "steered_best_rate": agg_metrics.get("steered_best_rate", 0),
            }
        
        # Store metrics
        metrics[experiment_name] = {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_processing_time": avg_processing_time,
            **steering_metrics
        }
    
    return metrics

def generate_comparison_table(metrics: Dict[str, Dict[str, float]]) -> str:
    """
    Generate a comparison table of metrics.
    
    Args:
        metrics: Dictionary mapping experiment names to metrics
        
    Returns:
        Formatted table as string
    """
    headers = ["Experiment"]
    metric_names = set()
    
    # Collect all metric names
    for experiment_metrics in metrics.values():
        metric_names.update(experiment_metrics.keys())
    
    # Sort metric names
    metric_names = sorted(metric_names)
    headers.extend(metric_names)
    
    # Create table rows
    rows = []
    for experiment_name, experiment_metrics in metrics.items():
        row = [experiment_name]
        for metric_name in metric_names:
            value = experiment_metrics.get(metric_name, "N/A")
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(value)
        rows.append(row)
    
    # Generate table
    return tabulate(rows, headers=headers, tablefmt="grid")

def plot_comparison(metrics: Dict[str, Dict[str, float]], output_path: str = None):
    """
    Generate comparison plots of metrics.
    
    Args:
        metrics: Dictionary mapping experiment names to metrics
        output_path: Path to save plot image (optional)
    """
    # Get metric names (excluding processing time for clarity)
    metric_names = set()
    for experiment_metrics in metrics.values():
        metric_names.update(experiment_metrics.keys())
    
    # Exclude processing time from plots
    if "avg_processing_time" in metric_names:
        metric_names.remove("avg_processing_time")
    
    metric_names = sorted(metric_names)
    
    # Setup plot
    num_metrics = len(metric_names)
    fig, axes = plt.subplots(1, num_metrics, figsize=(num_metrics * 4, 5))
    
    # Handle case with single metric
    if num_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        
        # Get values for this metric
        experiments = []
        values = []
        
        for experiment_name, experiment_metrics in metrics.items():
            if metric_name in experiment_metrics:
                experiments.append(experiment_name)
                values.append(experiment_metrics[metric_name])
        
        # Plot bars
        ax.bar(experiments, values)
        ax.set_title(metric_name)
        ax.set_ylim(0, max(values) * 1.2 if values else 1)
        ax.set_xticklabels(experiments, rotation=45, ha='right')
        
        # Add value labels on bars
        for j, v in enumerate(values):
            ax.text(j, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare results of multiple math500-with-steering experiments")
    parser.add_argument("result_files", nargs="+", help="Paths to result JSON files")
    parser.add_argument("--output", help="Path to save comparison report")
    parser.add_argument("--plot", help="Path to save comparison plot")
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.result_files)
    
    if not results:
        print("No valid results found!")
        return
    
    # Extract metrics
    metrics = extract_metrics(results)
    
    # Generate comparison table
    table = generate_comparison_table(metrics)
    print("\nComparison of Experiment Results:\n")
    print(table)
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("# Comparison of Experiment Results\n\n")
            f.write(table)
        print(f"\nComparison report saved to {args.output}")
    
    # Generate plots
    try:
        plot_comparison(metrics, args.plot)
    except Exception as e:
        print(f"Error generating plot: {str(e)}")

if __name__ == "__main__":
    main() 