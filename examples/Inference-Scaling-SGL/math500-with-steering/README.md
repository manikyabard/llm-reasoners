# Math500 with Steering Vectors

This project combines beam search for math problem solving with inference-time steering vectors. By recording model activations during generation and using preference signals to create steering vectors, we can guide the language model towards better reasoning without additional training.

## Overview

The implementation extends the standard beam search approach for math problem solving with the ability to compute and apply steering vectors at each step of the reasoning process. The key components are:

- **SteeringBeamSearch**: Extends BeamSearch to compute and apply steering vectors during search
- **SteeringMathModel**: Extends MathModel to support activation recording and steering vector application
- **SteeringMathConfig**: Search configuration for steering-enabled beam search

## YAML Configuration System

The project now includes a YAML configuration system for easier experimentation. Instead of specifying all parameters through command-line arguments, you can define and use YAML configuration files:

```bash
# Run with a specific configuration
python run_math500_steering.py --config configs/default.yaml

# Override specific settings from the command line
python run_math500_steering.py --config configs/default.yaml --num-examples 10
```

### Configuration Utilities

The project includes utilities to help manage configurations:

1. **run_experiments.sh**: A bash script to run experiments with different configurations:
   ```bash
   # Run with a specific configuration
   ./run_experiments.sh --config default

   # List available configurations
   ./run_experiments.sh --list

   # Run all configurations
   ./run_experiments.sh --all
   ```

2. **create_config.py**: A utility to create new configuration files:
   ```bash
   # Create a new configuration
   ./create_config.py my_config --description "My custom configuration"

   # Create a new configuration based on an existing one
   ./create_config.py my_custom_config --base high_temperature --description "Custom configuration based on high temperature"
   ```

For more details on the YAML configuration system, see [configs/README.md](configs/README.md).

## Two-Phase Action Generation

The implementation uses a two-phase approach for generating actions at each step:

1. **Initial Generation Phase**: 
   - Generate a set of initial candidate actions without steering
   - Compute rewards for these actions
   - Record activations and compute steering vectors based on these initial actions

2. **Steered Generation Phase**:
   - Apply the computed steering vectors to generate a new set of "steered" actions
   - Compute rewards for these steered actions
   - Combine initial and steered actions and select the best ones based on their rewards

This approach allows the model to refine its generations within the same step, using insights from the initial candidates to guide the generation of improved actions.

## Steering Effectiveness Metrics

The implementation includes comprehensive metrics tracking to evaluate the effectiveness of steering:

1. **Usage Metrics**:
   - **Steps with steering**: Tracks how many steps used steering vectors
   - **Steering usage rate**: Proportion of steps where steering was applied

2. **Selection Metrics**:
   - **Initial vs. steered selection**: Tracks how many actions from each source were selected
   - **Steered selection rate**: Proportion of selected actions that came from steered generation

3. **Reward Improvement Metrics**:
   - **Average reward before/after steering**: Compares rewards for initial vs. steered actions
   - **Mean reward improvement**: Average improvement in reward due to steering
   - **Steps with improvement**: Number of steps where steering improved the average reward

4. **Best Action Metrics**:
   - **Steered best rate**: How often the best action came from steered generation vs. initial generation
   - **Best action source tracking**: Source of the best action at each step

These metrics are automatically tracked during search and are included in the results output, making it easy to assess whether steering is providing meaningful improvements.

## Steering Strategies

The implementation supports multiple steering strategies:

1. **Reward Weighted** (`reward_weighted`): Weight activations by their corresponding rewards
2. **Best Only** (`best_only`): Use only the best candidate's activations
3. **Contrast** (`contrast`): Contrast best vs worst candidates
4. **No Steering** (`none`): Baseline without steering

## Running Experiments

You can run experiments with different steering strategies and configurations:

```bash
python run_math500_steering.py \
  --prompt-path path/to/your/prompts.json \
  --policy-model-path /path/to/your/policy_model \
  --reward-model-path /path/to/your/reward_model \
  --steering-strategy reward_weighted \
  --layers-to-use -1,-2,-3 \
  --components-to-use self_attn.o_proj \
  --num-actions 5 \
  --num-steered-actions 5 \
  --beam-size 3 \
  --max-depth 10 \
  --num-examples 5 \
  --output-path results.json
```

To run an ablation study comparing different steering strategies:

```bash
python run_math500_steering.py \
  --prompt-path path/to/your/prompts.json \
  --policy-model-path /path/to/your/policy_model \
  --reward-model-path /path/to/your/reward_model \
  --ablation-study \
  --num-examples 3 \
  --output-path ablation_results.json
```

## Configuration Options

### Model Settings
- `--policy-model-path`: Path to the policy model (default: "/data/manikya/Llama-3.2-1B-Instruct")
- `--reward-model-path`: Path to the reward model (default: "/data/manikya/math-shepherd-mistral-7b-prm")
- `--sglang-url`: Optional SGLang API URL for remote models

### Search Settings
- `--beam-size`: Beam size for search (default: 3)
- `--max-depth`: Maximum search depth (default: 10)
- `--temperature`: Temperature for sampling (default: 0.7)
- `--num-actions`: Number of initial actions to consider (default: 5)
- `--num-steered-actions`: Number of steered actions to generate (default: 5)

### Steering Settings
- `--steering-strategy`: Strategy for computing steering vectors (default: reward_weighted)
- `--steering-scale`: Scaling factor for steering vectors (default: 1.0)
- `--layers-to-use`: Comma-separated list of layers to use for steering (default: -1,-2,-3)
- `--components-to-use`: Comma-separated list of component paths (default: self_attn.o_proj)
- `--num-steering-candidates`: Number of candidates to use for steering computation (default: 5)
- `--combine-actions`: Whether to combine initial and steered actions (default: True)
- `--ablation-study`: Run ablation study comparing different steering strategies

### Experiment Settings
- `--num-examples`: Number of examples to process (default: 5)
- `--start-idx`: Starting index in the dataset (default: 0)
- `--output-path`: Path to save results (default: steering_results.json)
- `--device`: Device to run on (default: cuda:0)

## Implementation Details

### Two-Phase Generation Process

The implementation uses a novel two-phase approach for action generation and steering:

1. **Initial Generation**:
   - Generate N initial actions without any steering
   - Compute rewards for these actions
   - Record activations during generation

2. **Steering Vector Computation**:
   - Sort actions by reward
   - Select top K actions to compute steering vector
   - Apply the chosen steering strategy to compute steering vectors

3. **Steered Generation**:
   - Generate M new actions with the computed steering vectors
   - Apply steering during the generation process
   - Compute rewards for these steered actions

4. **Action Selection**:
   - Combine initial and steered actions
   - Sort by reward and select top actions for the beam

### Activation Recording

The implementation uses NNsight to record model activations during generation. These activations are recorded at specific layers and components specified in the configuration.

### Steering Vector Computation

Steering vectors are computed based on the recorded activations and their corresponding rewards. The strategy for computing steering vectors is configurable.

### Steering Vector Application

Steering vectors are applied to the model during generation by adding them to the activations at specified layers and components.

## Theoretical Underpinnings and Recent Advancements

This work builds on a growing body of research in activation engineering and mechanistic interpretability:

### Relationship to Recent Research

Recent advancements in mechanistic interpretability have shown that:

1. **Activation Steering**: Small interventions in model activations can significantly alter generation behavior while maintaining coherence.

2. **Preference-based Activation Engineering**: By using reward signals to guide activation modifications, we can steer models toward preferred behaviors without retraining.

3. **Layer-specific Interventions**: Certain layers (particularly later layers) have been shown to be more effective for steering without disrupting model coherence.

4. **Component Targeting**: Research has shown that targeting specific components (like attention outputs) can provide more efficient steering than modifying all activations.

### Practical Applications

The techniques implemented in this project have potential applications in:

- Improving reasoning capabilities in language models without fine-tuning
- Reducing hallucinations by steering away from low-confidence outputs
- Enhancing alignment through post-training interventions
- Creating task-specific variants of a base model without parameter updates

## File Structure

- `steering_beam_search.py`: Implementation of the steering-enabled beam search algorithm with metrics tracking
- `steering_math_model.py`: World model with support for steering
- `steering_config.py`: Configuration for steering-enabled search
- `run_math500_steering.py`: Main script to run experiments with comprehensive logging
- `configs/`: Directory containing YAML configuration files for different experiments
- `run_experiments.sh`: Script to run experiments with different configurations
- `create_config.py`: Utility to create new configuration files

## Requirements

- PyTorch
- Transformers
- NNsight
- Datasets
- Loguru
- tqdm
- PyYAML 