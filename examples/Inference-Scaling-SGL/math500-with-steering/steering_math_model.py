"""The steering-enabled world model for Math500.
This extends the regular MathModel to support activation recording and steering vectors.
"""

from typing import NamedTuple, Dict, List, Tuple, Any, Optional, Union
import torch
import copy
import re
import time
import nnsight
from loguru import logger

from reasoners import WorldModel, LanguageModel

MathAction = str


class MathState(NamedTuple):
    """The state of the Math problem.
    """
    step_idx: int
    steps: list
    end: bool
    solution: str


class SteeringMathModel(WorldModel):
    """Steering-enabled Math World Model
    Extends the standard MathModel to support activation recording and steering vectors.
    """

    def __init__(
        self,
        policy_model: nnsight.LanguageModel,
        reward_model: LanguageModel,
        prompt: dict,
        steering_config: Optional[Dict] = None,
        device: str = "cuda:0"
    ) -> None:
        """
        Initialize the SteeringMathModel.
        
        Args:
            policy_model: The NNsight-wrapped policy model for generation and recording activations
            reward_model: The reward model for evaluating solutions
            prompt: The prompt dictionary
            steering_config: Configuration for steering vectors
            device: Device to use for computation
        """
        super().__init__()
        
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.prompt = prompt
        self.device = device
        self.end = False
        self.question = None
        
        # Default steering configuration
        default_steering_config = {
            "layers_to_use": [-1, -2, -3],  # Default to last 3 layers
            "components_to_use": ["self_attn.o_proj"],  # Default to attention output
            "max_new_tokens": 400,
            "temperature": 0.7
        }
        
        # Override defaults with provided configuration
        self.steering_config = default_steering_config
        if steering_config:
            self.steering_config.update(steering_config)
            
        # Current steering vector to apply
        self.current_steering_vector = None
        
        # Initialize tokenizer
        self.tokenizer = self.policy_model.tokenizer

    def init_state(self) -> MathState:
        """Initialize the world model.

        :return: the initial state
        """
        return MathState(step_idx=0, steps=[], end=self.end, solution="")

    @staticmethod
    def step_helper(state: MathState, action_list: MathAction) -> Tuple[MathAction, bool, str]:
        """Helper function to process an action and extract relevant information.
        
        Args:
            state: Current state
            action_list: The action to process
            
        Returns:
            Tuple of (processed_action, is_terminal, solution)
        """
        step_num_to_take = state.step_idx + 1
        solution = state.solution

        if isinstance(action_list, list):
            action_list = action_list[0]
        if "## Step" in action_list:
            action = action_list.strip().split("## Step ")
        else:
            action = action_list.strip().split("Step ")

        num_actions = len(action)
        
        # Use regex to extract the number right after "Step" word in the last action
        last_action_step_num = int(re.findall(r"\d+", action[-1])[0])
        try:
            first_action_step_num = int(re.findall(r"\d+", action[0])[0])
        except IndexError:
            first_action_step_num = int(re.findall(r"\d+", action[1])[0])

        if num_actions != last_action_step_num - first_action_step_num + 1:
            # There is a potential parsing error.
            # The last action should have the step number equal to the number of actions.
            all_actions = []
            for a in action:
                all_actions += a.split("Step ")
            action = all_actions

        is_terminal = False

        # Check if this is the terminal step with a solution
        future_actions_present = (
            len([a for a in action if a.strip().startswith(f"{step_num_to_take + 1}")])
            > 0
        )

        if ("$\\boxed{" in action[-1]) and not future_actions_present:
            solution = action[-1].split("$\\boxed{")
            solution = solution[-1].split("}$")
            solution = solution[0]
            is_terminal = True

        action_to_take = action[-1]

        # Fix step numbering if needed
        if action_to_take.startswith("1:"):
            action_to_take = action_to_take.replace("1:", f"{step_num_to_take}:", 1)

        return action_to_take, is_terminal, solution

    def step(self, state: MathState, action_list: MathAction) -> tuple[MathState, dict]:
        """Take a step in the world model.

        Args:
            state: The current state
            action_list: The action to take
            
        Returns:
            The next state and additional information cached for reward calculation
        """
        start = time.time()
        action_to_take, is_terminal, solution = self.step_helper(state, action_list)

        logger.info(f"Action to take at step {state.step_idx+1}:\n{action_to_take}")

        state = copy.deepcopy(state)
        steps = state.steps + [action_to_take]
        end = is_terminal

        state = MathState(
            step_idx=state.step_idx + 1,
            steps=steps,
            end=end,
            solution=solution,
        )

        logger.info(f"TIME: function step for Step {state.step_idx} took {time.time()-start} seconds")

        return state, {}

    def is_terminal(self, state: MathState) -> bool:
        """Check if the state is terminal."""
        return state.end
        
    def prepare_prompt(self, state: MathState) -> str:
        """Prepare prompt for generation.
        
        Args:
            state: Current state
            
        Returns:
            Formatted prompt for the model
        """
        problem_state = (
            "## Step " + "\n\n## Step ".join([f"{step}" for step in state.steps])
            if len(state.steps) != 0
            else ""
        )

        return (
            self.prompt["icl"]
            .replace("<init_state>", self.question["init"])
            .replace("<problem_state>", problem_state)
        )
    
    def generate_with_steering(self, prompt: str, steering_vector: Optional[Dict] = None) -> Tuple[str, Dict]:
        """Generate text using the policy model with steering vector application.
        
        Args:
            prompt: The prompt to use for generation
            steering_vector: Optional steering vector to apply during generation
            
        Returns:
            Generated text and a dictionary of activations
        """
        activation_data = {}
        
        layers_to_use = self.steering_config["layers_to_use"]
        components_to_use = self.steering_config["components_to_use"]
        
        # Ensure layers_to_use has proper indices
        num_layers = len(self.policy_model.model.layers)
        layers_to_use = [layer if layer >= 0 else num_layers + layer for layer in layers_to_use]
        
        # Generate with nnsight
        with self.policy_model.generate(
            prompt, 
            max_new_tokens=self.steering_config.get("max_new_tokens", 400), 
            temperature=self.steering_config.get("temperature", 0.7)
        ) as generator:
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
                    
                    # Apply steering vector if provided
                    if steering_vector is not None and layer_name in steering_vector:
                        component.output = component.output + steering_vector[layer_name]
                        logger.debug(f"Applied steering to {layer_name} with shape {steering_vector[layer_name].shape}")
            
            # Get the generated tokens
            tokens = self.policy_model.generator.output.save()
        
        # Decode tokens to text
        generation = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        
        return generation, activation_data
    
    def set_question(self, question: Dict[str, str]) -> None:
        """Set the current question.
        
        Args:
            question: The question to process
        """
        self.question = question
        
    def set_steering_vector(self, steering_vector: Dict) -> None:
        """Set the current steering vector.
        
        Args:
            steering_vector: The steering vector to use
        """
        self.current_steering_vector = steering_vector 