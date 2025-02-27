"""
Search configuration for steering-enabled beam search.
This extends the standard MathConfig to support steering vector application.
"""

import numpy as np
import time
import torch
import torch.nn.functional as F
import nnsight
from typing import List, Dict, Tuple, Any, Optional, Union

from loguru import logger

from steering_math_model import MathState, MathAction, SteeringMathModel
from reasoners import SearchConfig, LanguageModel


class SteeringMathConfig(SearchConfig):
    """
    Search configuration for steering-enabled beam search in math problem solving.
    This extends the standard MathConfig to support activation recording and steering vectors.
    """
    
    def __init__(
        self,
        policy_model: nnsight.LanguageModel,
        reward_model: LanguageModel,
        prompt: dict,
        steering_config: Optional[Dict] = None,
        batch_size: int = 8,
        num_actions: int = 3,
        temperature: float = 0.7,
        device: str = "cuda:0"
    ) -> None:
        """
        Initialize the SteeringMathConfig.
        
        Args:
            policy_model: The NNsight-wrapped policy model for generation and recording activations
            reward_model: The reward model for evaluating solutions
            prompt: The prompt dictionary
            steering_config: Configuration for steering vectors
            batch_size: Batch size for generation
            num_actions: Number of actions to generate
            temperature: Temperature for generation
            device: Device to use for computation
        """
        super().__init__()
        
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.temperature = temperature
        self.device = device
        
        # Default steering configuration
        default_steering_config = {
            "layers_to_use": [-1, -2, -3],  # Default to last 3 layers
            "components_to_use": ["self_attn.o_proj"],  # Default to attention output
            "max_new_tokens": 400,
            "temperature": self.temperature,
            "use_steering": True,
            "num_candidates_to_keep": min(5, self.num_actions),  # Number of candidates to keep for steering
            "num_steered_generations": min(5, self.num_actions)  # Number of steered generations to produce
        }
        
        # Override defaults with provided configuration
        self.steering_config = default_steering_config
        if steering_config:
            self.steering_config.update(steering_config)
            
        # Current steering vector to apply during generation
        self.current_steering_vector = None
            
        # Initialize tokenizer
        self.tokenizer = self.policy_model.tokenizer
        
    def get_actions(self, state: MathState) -> List[MathAction]:
        """
        Get actions from the current state without recording activations.
        This is a simplified version that doesn't record activations.
        
        Args:
            state: Current state
            
        Returns:
            List of actions
        """
        actions, _ = self.get_actions_with_activations(state)
        return actions
        
    def get_actions_with_activations(
        self, 
        state: MathState, 
        num_actions: Optional[int] = None
    ) -> Tuple[List[MathAction], List[Dict]]:
        """
        Get actions and their activations from the current state.
        
        Args:
            state: Current state
            num_actions: Number of actions to generate (overrides self.num_actions if provided)
            
        Returns:
            Tuple of (actions, activations)
        """
        start_time = time.time()
        
        # Use provided num_actions or fall back to self.num_actions
        n_actions = num_actions if num_actions is not None else self.num_actions
        
        # Check if we are using steering
        steering_active = self.current_steering_vector is not None and self.steering_config.get("use_steering", True)
        if steering_active:
            logger.debug(f"Generating {n_actions} actions with steering at step {state.step_idx}")
        else:
            logger.debug(f"Generating {n_actions} actions without steering at step {state.step_idx}")
        
        # Prepare prompt
        problem_state = (
            "## Step " + "\n\n## Step ".join([f"{step}" for step in state.steps])
            if len(state.steps) != 0
            else ""
        )

        prompts = (
            self.prompt["icl"]
            .replace("<init_state>", self.example["init"])
            .replace("<problem_state>", problem_state)
        )
        
        actions = []
        activations_list = []
        
        layers_to_use = self.steering_config["layers_to_use"] 
        components_to_use = self.steering_config["components_to_use"]
        
        # Ensure layers_to_use has proper indices
        num_layers = len(self.policy_model.model.layers)
        layers_to_use = [layer if layer >= 0 else num_layers + layer for layer in layers_to_use]
        
        # Define a stopping string
        stop_string = f"## Step {state.step_idx + 2}"
        
        # Create a custom stopping criteria based on the stop string
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        class StopStringCriteria(StoppingCriteria):
            def __init__(self, tokenizer, stop_string, prompt):
                self.tokenizer = tokenizer
                self.stop_string = stop_string
                self.prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
                self.prompt_len = len(self.prompt_ids)
                
            def __call__(self, input_ids, scores, **kwargs):
                # Get the generated text (excluding the prompt)
                if input_ids.shape[1] <= self.prompt_len:
                    return False
                
                # Only check the last 50 tokens for efficiency
                last_tokens = input_ids[0, -min(50, input_ids.shape[1]):]
                
                try:
                    # Decode the sequence (the last part that might contain the stop string)
                    generated_text = self.tokenizer.decode(last_tokens)
                    
                    # Check if stop string is in the generated text
                    return self.stop_string in generated_text
                except Exception as e:
                    # If we encounter any error, log it and continue generation
                    logger.warning(f"Error in stopping criteria: {str(e)}")
                    return False
        
        # Create the stopping criteria
        stopping_criteria = StoppingCriteriaList([
            StopStringCriteria(self.tokenizer, stop_string, prompts)
        ])
        
        # Generate multiple actions with activation recording
        for i in range(n_actions):
            # Generate with nnsight
            with self.policy_model.generate(
                prompts, 
                max_new_tokens=self.steering_config.get("max_new_tokens", 400),
                temperature=self.temperature,
                do_sample=True,
                stopping_criteria=stopping_criteria
            ) as generator:
                # Track activations
                activation_data = {}
                
                # Apply .all() to model to streamline interventions across all token generations
                self.policy_model.all()
                
                # Register hooks for specified layers and components
                for layer_idx in layers_to_use:
                    # Get the layer
                    layer = self.policy_model.model.layers[layer_idx]
                    
                    # Register hooks for each component
                    for component_path in components_to_use:
                        # Navigate the component path
                        component = layer
                        for part in component_path.split("."):
                            component = getattr(component, part)
                        
                        # Store activations for computing steering vectors
                        layer_name = f"model.layers.{layer_idx}.{component_path}"
                        activation_data[layer_name] = component.output.save()
                        
                        # Apply steering vector if provided and enabled
                        if steering_active and layer_name in self.current_steering_vector:
                            component.output = component.output + self.current_steering_vector[layer_name]
                            logger.debug(f"Applied steering to {layer_name} at position {i+1}/{n_actions}")
                
                # Get the generated tokens
                tokens = self.policy_model.generator.output.save()
            
            # Decode tokens to text
            generation = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
            
            # Process the generated text to extract the action
            action = generation.replace(stop_string, "").strip()
            
            # Store action and activations
            actions.append(action)
            activations_list.append(activation_data)

        logger.debug(f"Generated {len(actions)} actions at step {state.step_idx} in {time.time() - start_time:.2f}s")
        if steering_active:
            logger.debug(f"Used steering vectors for generation with strategy: {self.steering_config.get('steering_strategy', 'unknown')}")
        
        return actions, activations_list
    
    def reward(
        self,
        state: MathState,
        action: MathAction,
        intuition: float = None
    ) -> float:
        """
        Calculate reward for a given action.
        
        Args:
            state: Current state
            action: Action to evaluate
            intuition: Optional pre-computed intuition
            
        Returns:
            Reward value and auxiliary information
        """
        # Define tokens for the reward model
        good_token = "+"
        bad_token = "-"
        step_tag = "ки"

        # Prepare the current problem state
        current_problem_state = "\n".join(
            [f"Step {step.strip()} {step_tag}" for step in state.steps]
        )

        # Process the action to match the expected format
        action_to_take, _, _ = SteeringMathModel.step_helper(state, action)

        # Add the current action to the problem state
        current_problem_state += (
            f"\nStep {state.step_idx + 1} {action_to_take} {step_tag}"
        )

        # Create the input for the reward model
        input_for_prm = f"{self.example['init']} {current_problem_state}"

        input_id = torch.tensor([self.reward_model.tokenizer.encode(input_for_prm)])

        candidate_tokens = self.reward_model.tokenizer.encode(f"{good_token} {bad_token}")[1:]
        step_tag_id = self.reward_model.tokenizer.encode(f"{step_tag}")[-1]

        with torch.no_grad():
            logits = self.reward_model.model(input_id).logits[:, :, candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0] 
            step_scores = scores[input_id == step_tag_id]
        
        intuition = step_scores[-1].item()
        
        logger.debug(
            f"Reward for step {state.step_idx} is: {intuition} where the potential step is: {action_to_take}"
        )

        return intuition, {"intuition": intuition}
    
    def fast_reward(self, state: MathState, action: MathAction) -> Tuple[float, Dict]:
        """
        Calculate a fast approximation of the reward.
        This is used for pruning candidates before computing the full reward.
        
        Args:
            state: Current state
            action: Action to evaluate
            
        Returns:
            Fast reward value and auxiliary information
        """
        # For now, we don't have a fast approximation, so return 0.0
        return 0.0, {}
    
    def set_steering_vector(self, steering_vector: Dict) -> None:
        """
        Set the current steering vector.
        
        Args:
            steering_vector: Steering vector to use for subsequent generations
        """
        self.current_steering_vector = steering_vector
        if steering_vector is None:
            logger.debug("Cleared steering vector")
        else:
            logger.debug(f"Set new steering vector with {len(steering_vector)} layer entries")
        
    def set_example(self, example: Dict[str, str]) -> None:
        """
        Set the current example.
        
        Args:
            example: Example to use for generation
        """
        self.example = example 