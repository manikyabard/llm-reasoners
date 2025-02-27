# YAML Configuration System for Math500-with-Steering

This directory contains YAML configuration files for running different experiments with the Math500-with-Steering system. The YAML configuration system makes it easy to define and run various experimental setups without modifying the code.

## Usage

To run an experiment with a YAML configuration file:

```bash
python run_math500_steering.py --config configs/default.yaml
```

You can also override specific configuration values from the command line:

```bash
python run_math500_steering.py --config configs/default.yaml --num-examples 10
```

## Configuration Structure

Each configuration file is organized into the following sections:

### model_settings

Settings related to the language models used for the experiment.

- `policy_model_path`: Path to the policy model
- `reward_model_path`: Path to the reward model
- `sglang_url`: Optional SGLang API URL for remote models
- `device`: Device to run on (default: "cuda:0")

### search_settings

Settings for the beam search algorithm.

- `beam_size`: Beam size for search
- `max_depth`: Maximum search depth
- `temperature`: Temperature for sampling
- `num_actions`: Number of initial actions to consider
- `num_steered_actions`: Number of steered actions to generate

### steering_settings

Settings for the steering vectors.

- `steering_strategy`: Strategy for computing steering vectors ("reward_weighted", "best_only", "contrast", "none")
- `steering_scale`: Scaling factor for steering vectors
- `layers_to_use`: List of layers to use for steering (can be negative indices)
- `components_to_use`: List of component paths to use for steering
- `num_steering_candidates`: Number of candidates to use for steering computation
- `combine_actions`: Whether to combine initial and steered actions

### experiment_settings

General experiment settings.

- `prompt_path`: Path to prompts JSON file
- `output_path`: Path to save results
- `log_file`: Path to log file
- `num_examples`: Number of examples to process
- `start_idx`: Starting index in the dataset
- `num_processes`: Number of parallel processes
- `ablation_study`: Whether to run an ablation study comparing different steering strategies

## Available Configurations

1. **default.yaml**: Baseline configuration with standard settings.
2. **high_temperature.yaml**: Configuration with higher temperature for more diverse exploration.
3. **ablation_study.yaml**: Configuration for running an ablation study comparing different steering strategies.
4. **component_test.yaml**: Configuration for testing steering with different model components.

## Creating Your Own Configuration

To create a new configuration, copy one of the existing files and modify it to suit your needs. Ensure that all required fields are present in each section.

Example:

```yaml
# My custom configuration

model_settings:
  policy_model_path: "/path/to/my/policy_model"
  reward_model_path: "/path/to/my/reward_model"
  device: "cuda:0"

search_settings:
  beam_size: 5
  max_depth: 15
  temperature: 0.8
  num_actions: 6
  num_steered_actions: 6

steering_settings:
  steering_strategy: "best_only"
  steering_scale: 1.2
  layers_to_use: [-1, -2, -3]
  components_to_use: ["self_attn.o_proj"]
  num_steering_candidates: 5
  combine_actions: true

experiment_settings:
  prompt_path: "path/to/prompts.json"
  output_path: "results/my_custom_results.json"
  log_file: "logs/my_custom_run.log"
  num_examples: 10
  start_idx: 0
  num_processes: 1
``` 