"""
Proof of concept for dynamic steering vectors in math problem solving.
Uses NNsight to analyze model activations and create steering vectors based on PRM scores.
"""

from sympy import solve
import torch
import nnsight
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from typing import List, Dict, Tuple, Union, Any
from loguru import logger
import torch.nn.functional as F
import re

class SteeringPOC:
    def __init__(
        self,
        policy_model_path: str = "/data/manikya/Llama-3.2-1B-Instruct",   
        reward_model_path: str = "/data/manikya/Llama3.1-8B-PRM-Deepseek-Data",  
        policy_model=None,
        reward_model=None,
        device: str = "auto",
        num_generations: int = 2,  # Reduced for testing
        temperature: float = 0.7,
        layers_to_use: List[int] = None,  # Allow specifying which layers to use
        components_to_use: List[str] = None,  # Allow specifying which components to use
        layer_weights: Dict[int, float] = None,  # Allow specifying weights for each layer
        component_weights: Dict[str, float] = None,  # Allow specifying weights for each component
        steering_scale: float = 1.0,  # Global scaling factor for steering vectors
        debug: bool = False
    ):
        # Initialize models with nnsight
        if policy_model is None:
            self.policy_model = nnsight.LanguageModel(policy_model_path, device_map=device)
        else:
            self.policy_model = policy_model
        if reward_model is None:
            self.reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path).eval().to(device)
        else:
            self.reward_model = reward_model
        
        # Initialize tokenizers
        self.policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_path)
        # self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
        self.reward_tokenizer = LlamaTokenizer.from_pretrained(reward_model_path)

        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = 'ки'
        
        # Move candidate tokens to correct device
        self.candidate_tokens = torch.tensor(
            self.reward_tokenizer.encode(f"{self.good_token} {self.bad_token}")[1:],
            device=device
        )
        self.step_tag_id = torch.tensor(
            self.reward_tokenizer.encode(f"{self.step_tag}")[-1],
            device=device
        )
        
        # Store config
        self.device = device
        self.num_generations = num_generations
        self.temperature = temperature

        # Handle layers and components configuration with defaults
        num_layers = len(self.policy_model.model.layers)
        self.layers_to_use = layers_to_use or [-1, -2, -3]  # Default to last 3 layers
        self.components_to_use = components_to_use or ["self_attn.o_proj"]  # Default to attention output projection
        
        # Convert negative indices to positive
        self.layers_to_use = [layer if layer >= 0 else num_layers + layer for layer in self.layers_to_use]
        
        # Set layer and component weights
        self.layer_weights = layer_weights or {layer: 1.0 for layer in self.layers_to_use}
        self.component_weights = component_weights or {component: 1.0 for component in self.components_to_use}
        self.steering_scale = steering_scale
        
        # Hook storage
        self.activation_data = {}
        
        # Debugging flag
        self.debug = debug

    def prepare_prompt(self, problem: str, target_step: int = None) -> str:
        """Prepare prompt for generation.
        
        Args:
            problem: The math problem to solve
            target_step: If provided, only generate this specific step. Otherwise generate full solution.
        """
        template = f"""solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.

{problem}"""

        return template

    def get_reward(self, problem: str, steps_list: List[str] = None, target_step: int = None) -> float:
        """Calculate reward for a given step using the PRM."""
        steps_list = [f"{step.strip()} {self.step_tag}" for step in steps_list]
        steps_str = '\n'.join(steps_list)
        input_for_prm = f"{problem} {steps_str}"
        
        # Create tensor and move to correct device
        input_id = torch.tensor([self.reward_tokenizer.encode(input_for_prm)], device=self.device)

        with torch.no_grad():
            logits = self.reward_model(input_id).logits[:, :, self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0] 
            step_scores = scores[input_id == self.step_tag_id]

        return step_scores[-1] if target_step is None else step_scores[target_step-1]

    def compute_steering_vector(self, activations_list: List[Dict], rewards: List[float]) -> Dict:
        """Compute steering vectors based on activations and their corresponding rewards.
        
        Args:
            activations_list: List of dictionaries containing activations from different generations
            rewards: List of corresponding rewards for each generation
            
        Returns:
            Dictionary of steering vectors for each layer
        """
        # Normalize rewards to weights - move to same device as activations
        weights = F.softmax(torch.tensor(rewards, device=self.device) / self.temperature, dim=0)
        
        steering_vectors = {}
        for layer_name in activations_list[0].keys():
            # Parse layer path to get layer index and component
            parts = layer_name.split(".")
            layer_idx = int(parts[2])
            component = ".".join(parts[3:])
            
            # Skip if this layer or component is not in our configured list
            if layer_idx not in self.layers_to_use or component not in self.components_to_use:
                continue
                
            # Get layer and component weights
            layer_weight = self.layer_weights.get(layer_idx, 1.0)
            component_weight = self.component_weights.get(component, 1.0)
            
            # Stack activations for this layer from all generations
            # We take only the first activation tensor as it contains the prompt activations
            # which is what we need for steering
            layer_activations = torch.stack([act[layer_name] for act in activations_list])
            
            # Compute weighted sum based on rewards
            weighted_sum = torch.sum(layer_activations * weights.view(-1, 1, 1, 1), dim=0)
            
            # Compute mean activation
            mean_activation = torch.mean(layer_activations, dim=0)
            
            # Compute steering vector as the difference between weighted sum and mean
            # Apply layer, component, and global scaling factors
            steering_vectors[layer_name] = (weighted_sum - mean_activation) * layer_weight * component_weight * self.steering_scale
        
        return steering_vectors

    def generate(self, problem: str, target_step: int = None, steering_vector: Dict = None) -> str:
        """Generate text using the policy model.
        
        Args:
            problem: The problem to solve
            target_step: Which step to focus on (if None, generates full solution)
            steering_vector: Optional steering vector to apply during generation
            
        Returns:
            Generated text
        """
        # Prepare prompt
        prompt = self.prepare_prompt(problem, target_step)
        
        # Generate with nnsight
        with self.policy_model.generate(prompt, max_new_tokens=400, temperature=self.temperature) as generator:
            # Initialize activation data
            self.activation_data = {}
            
            # Apply .all() to model to streamline interventions across all token generations
            self.policy_model.all()
            
            # Register hooks for specified layers and components
            for layer_idx in self.layers_to_use:
                # Get the layer
                layer = self.policy_model.model.layers[layer_idx]
                
                # Register hooks for each component
                for component_path in self.components_to_use:
                    # Navigate the component path
                    component = layer
                    for part in component_path.split("."):
                        component = getattr(component, part)
                    
                    # Store activations for computing steering vectors
                    layer_name = f"model.layers.{layer_idx}.{component_path}"
                    self.activation_data[layer_name] = component.output.save()
                    
                    # Apply steering vector if provided
                    if steering_vector is not None and layer_name in steering_vector:
                        component.output = component.output + steering_vector[layer_name]
                        if self.debug:
                            print(f"Applied steering to {layer_name} with shape {steering_vector[layer_name].shape}")
                    elif steering_vector is not None and layer_name not in steering_vector:
                        if self.debug:
                            print(f"WARNING: {layer_name} not found in steering_vector keys: {list(steering_vector.keys())}")
            
            # Get the generated tokens
            tokens = self.policy_model.generator.output.save()
        
        # Decode tokens to text
        generation = self.policy_tokenizer.decode(tokens[0], skip_special_tokens=True)
        
        return generation, tokens[0]

    def extract_steps(self, text: str, target_step: int = None) -> Union[str, List[str]]:
        """Improved step extraction with regex patterns that ignore template examples"""
        # Match lines starting with ## Step followed by number, handling various formats
        step_pattern = re.compile(r'^\s*##\s*Step\s*(\d+):?([^[]*$|.*$)', re.IGNORECASE | re.MULTILINE)
        steps = []
        
        # Find all matches in text
        matches = list(step_pattern.finditer(text))
        
        if not matches:
            logger.warning("No steps found in text using regex pattern")
            # Check for boxed answer only
            final_answer = re.search(r'\\boxed{.*}', text)
            if final_answer:
                logger.debug(f"Found final answer: {final_answer.group()}")
                steps.append(final_answer.group())
            return steps

        # Process matches and extract step content
        for i, match in enumerate(matches):
            step_num = int(match.group(1))
            current_start = match.start()
            
            # Find the end of this step (start of next step or end of text)
            if i < len(matches) - 1:
                next_start = matches[i+1].start()
                step_content = text[current_start:next_start]
            else:
                step_content = text[current_start:]
            
            steps.append(step_content.strip())
        
        # Handle final answer separately
        final_answer = re.search(r'\\boxed{.*}', text)
        if final_answer:
            # Add final answer only if not already present in last step
            if steps and final_answer.group() not in steps[-1]:
                steps.append(final_answer.group())
        
        if target_step is not None:
            for step in steps:
                step_num_match = re.search(r'##\s*Step\s*(\d+)', step)
                if step_num_match and int(step_num_match.group(1)) == target_step:
                    return step
            return ""  # Target step not found
        
        return steps

    def filter_activations_for_target_step(self, tokens, completion_text, target_step):
        """Filter activations to only include those up to the target step.
        
        Args:
            tokens: The generated token IDs
            completion_text: The decoded completion text
            target_step: The target step number
            
        Returns:
            Filtered activation data
        """
        if target_step is None:
            return self.activation_data
            
        # Extract the steps
        steps = self.extract_steps(completion_text)
        
        if not steps or len(steps) < target_step:
            # If we don't have enough steps, return all activations
            return self.activation_data
            
        # Join steps up through the target step
        steps_text = "\n".join(steps[:target_step])
        
        # Count tokens in the relevant part of the text
        step_tokens = self.policy_tokenizer.encode(steps_text)
        token_count = len(step_tokens)
        
        # Filter activations to only include tokens up to the target step
        filtered_activations = {}
        for layer_name, activations in self.activation_data.items():
            # Keep only activations for tokens up to the target step
            # The +1 is to include the token that generates the end of the target step
            filtered_activations[layer_name] = activations[:, :token_count+1, :]
            
        return filtered_activations

    def run_poc(self, problem: str, target_step: int = None) -> Dict[str, Any]:
        """Run the proof of concept pipeline.
        
        Args:
            problem: The problem to solve
            target_step: Which step to focus on (if None, generates full solution)
        """
        # Generate multiple completions
        generations = []
        rewards = []
        activations_list = []
        all_generations = []
        
        for i in range(self.num_generations):
            # Clear previous activation data
            self.activation_data = {}
            
            # Generate completion
            completion, tokens = self.generate(problem, target_step)
            logger.debug(f"Generation {i+1}: {completion}")
            all_generations.append(completion)
            
            # Extract steps from completion
            steps = self.extract_steps(completion)
            if not steps:
                logger.warning(f"No steps found in generation {i+1}")
                continue
                
            # Get reward for target step
            if target_step is not None:
                if target_step > len(steps):
                    logger.warning(f"Target step {target_step} not found in generation {i+1}")
                    continue
                    
                # Filter activations to only include those up to the target step
                filtered_activations = self.filter_activations_for_target_step(tokens, completion, target_step)
                
                # Store filtered activations and successful generation
                activations_list.append({k: v.clone() for k, v in filtered_activations.items()})
                generations.append(completion)
                
                # Get reward for target step
                reward = self.get_reward(problem, steps, target_step)
                rewards.append(reward)
            else:
                # If no target step, evaluate the first step and use all activations
                activations_list.append({k: v.clone() for k, v in self.activation_data.items()})
                generations.append(completion)
                reward = self.get_reward(problem, steps, 1)
                rewards.append(reward)
        
        # Check if we have enough valid generations
        if len(rewards) == 0:
            logger.warning(f"No valid generations found with target step {target_step}")
            return {
                "original_generations": all_generations,
                "original_rewards": [0.0] * len(all_generations),
                "steered_generation": "",
                "steered_reward": 0.0,
                "activations_list": []
            }
            
        if len(rewards) < 2:
            logger.warning(f"Only {len(rewards)} valid generation found with target step {target_step}, need at least 2 for steering")
            return {
                "original_generations": generations,
                "original_rewards": rewards,
                "steered_generation": generations[0] if generations else "",
                "steered_reward": rewards[0] if rewards else 0.0,
                "activations_list": activations_list
            }
            
        # Compute steering vector based on activations and rewards
        steering_vectors = self.compute_steering_vector(activations_list, rewards)
        
        # Generate steered completion
        steered_completion, _ = self.generate(problem, target_step, steering_vectors)
        
        # Get reward for steered completion
        steered_steps = self.extract_steps(steered_completion)
        if steered_steps:
            if target_step is not None:
                if target_step <= len(steered_steps):
                    steered_reward = self.get_reward(problem, steered_steps, target_step)
                else:
                    steered_reward = 0.0
            else:
                steered_reward = self.get_reward(problem, steered_steps, 1)
        else:
            steered_reward = 0.0
            
        return {
            "original_generations": generations,  # Only include valid generations
            "original_rewards": rewards,          # Only include rewards for valid generations
            "steered_generation": steered_completion,
            "steered_reward": steered_reward,
            "activations_list": activations_list  # Return the list of activations for valid generations
        }

def test_all_method():
    """Test function to verify the behavior of .all() method for token-by-token generation."""
    print("Testing .all() method for token-by-token generation...")
    
    # Initialize a small model for testing
    model = nnsight.LanguageModel("/data/manikya/Llama-3.2-1B-Instruct", device_map="auto")
    
    prompt = "The capital of France is"
    n_new_tokens = 5
    print(f"Using prompt: '{prompt}'")
    
    print("\nTesting with regular generation:")
    with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
        # Get the generated tokens
        tokens = model.generator.output.save()
    
    generation = model.tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(f"Regular generation: '{generation}'")
    
    print("\nTesting with .all() method:")
    with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
        # Initialize an nnsight list to store activations
        activation_list = nnsight.list().save()
        
        # Apply .all() to model
        model.all()
        
        # Get activation from last layer for each token
        last_layer = model.model.decoder.layers[-1]
        
        # Use append to capture activations for each token
        activation_list.append(last_layer.self_attn.v_proj.output)
        
        # Get the generated tokens
        tokens = model.generator.output.save()
    
    generation = model.tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(f"Generation with .all(): '{generation}'")
    print(f"Number of activations captured: {len(activation_list)}")
    
    print("\nNow testing with .next() for comparison:")
    with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
        # Get first token activation
        last_layer = model.model.decoder.layers[-1]
        activation1 = last_layer.self_attn.v_proj.output.save()
        
        # Get second token activation
        activation2 = last_layer.self_attn.v_proj.next().output.save()
        
        # Get third token activation
        activation3 = last_layer.self_attn.v_proj.next().output.save()
        
        # Get the rest of the tokens
        for _ in range(n_new_tokens - 3):
            last_layer.self_attn.v_proj.next()
        
        # Get the generated tokens
        tokens = model.generator.output.save()
    
    generation = model.tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(f"Generation with .next(): '{generation}'")
    print(f"Activation1 shape: {activation1.shape}, Activation2 shape: {activation2.shape}, Activation3 shape: {activation3.shape}")
    
    print("\nTest completed successfully!")
    return activation_list

if __name__ == "__main__":
    # Run the simplified test
    activations = test_all_method()