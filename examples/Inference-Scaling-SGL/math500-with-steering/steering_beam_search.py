"""
Steering-enabled beam search algorithm for math problem solving.
This combines beam search with steering vectors to improve reasoning at inference time.
"""

from typing import Generic, Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
import nnsight
from copy import deepcopy

from reasoners.algorithm.beam_search import BeamSearch, BeamSearchNode, BeamSearchResult
from reasoners import SearchAlgorithm, WorldModel, SearchConfig, State, Action

class SteeringBeamSearch(BeamSearch):
    """
    A beam search algorithm that applies steering vectors during generation.
    This extends the standard beam search to include steering vector computation
    and application at each step.
    """
    
    def __init__(
        self, 
        beam_size: int, 
        max_depth: int,
        steering_config: Dict = None,
        **kwargs
    ):
        """
        Initialize the SteeringBeamSearch algorithm.
        
        Args:
            beam_size: The number of candidates to keep at each step
            max_depth: The maximum number of steps to take
            steering_config: Configuration for steering vectors, including:
                - layers_to_use: List of model layers to apply steering to
                - components_to_use: List of component paths within layers
                - steering_scale: Scaling factor for steering vectors
                - layer_weights: Optional weights for each layer
                - component_weights: Optional weights for each component 
                - steering_strategy: How to compute steering vectors
                    - "reward_weighted" (default): Weight by reward
                    - "best_only": Use only the best candidate
                    - "contrast": Contrast best vs worst
                - num_steering_candidates: How many candidates to use for steering
                - num_steered_generations: Number of steered generations to produce
            **kwargs: Additional arguments for the BeamSearch algorithm
        """
        super().__init__(beam_size, max_depth, **kwargs)
        
        # Default steering configuration
        default_steering_config = {
            "layers_to_use": [-1, -2, -3],  # Default to last 3 layers
            "components_to_use": ["self_attn.o_proj"],  # Default to attention output
            "steering_scale": 1.0,
            "layer_weights": None,
            "component_weights": None,
            "steering_strategy": "reward_weighted",
            "num_steering_candidates": 5,  # Number of candidates to use for steering
            "num_steered_generations": 5,  # Number of steered generations to produce
            "combine_initial_and_steered": True  # Whether to combine initial and steered actions
        }
        
        # Override defaults with provided configuration
        self.steering_config = default_steering_config
        if steering_config:
            self.steering_config.update(steering_config)
            
        # Initialize tracking variables for activations and steering vectors
        self.activation_storage = {}
        self.current_steering_vector = None
        
        # Metrics tracking for evaluating steering effectiveness
        self.steering_metrics = {
            "steps_with_steering": 0,
            "total_steps": 0,
            "initial_actions_selected": 0,
            "steered_actions_selected": 0,
            "initial_reward_avg": [],
            "steered_reward_avg": [],
            "avg_reward_improvement": [],
            "steps_with_steered_improvement": 0,
            "best_action_source": []  # "initial" or "steered" for each step
        }
        
    def compute_steering_vector(
        self, 
        activations_list: List[Dict], 
        rewards: List[float],
        device: str = "cuda:0"
    ) -> Dict:
        """
        Compute steering vectors based on activations and their corresponding rewards.
        
        Args:
            activations_list: List of dictionaries containing activations from different generations
            rewards: List of corresponding rewards for each generation
            device: Device to use for computation
            
        Returns:
            Dictionary of steering vectors for each layer
        """
        if not activations_list or len(activations_list) < 2:
            logger.warning("Not enough activations to compute steering vector")
            return {}
            
        # Strategy-specific processing
        strategy = self.steering_config["steering_strategy"]
        
        if strategy == "best_only" and len(activations_list) > 1:
            # Use only the best candidate's activations
            best_idx = np.argmax(rewards)
            best_activation = activations_list[best_idx]
            # Compute steering as deviation of best from mean of all
            activations_list_without_best = [
                act for i, act in enumerate(activations_list) if i != best_idx
            ]
            return self._compute_steering_vector_contrast(
                [best_activation], 
                activations_list_without_best,
                device
            )
            
        elif strategy == "contrast" and len(activations_list) > 1:
            # Contrast best vs worst
            sorted_idx = np.argsort(rewards)
            best_activations = [activations_list[sorted_idx[-1]]]
            worst_activations = [activations_list[sorted_idx[0]]]
            return self._compute_steering_vector_contrast(
                best_activations, 
                worst_activations,
                device
            )
            
        else:  # Default: "reward_weighted"
            # Weight activations by normalized rewards
            return self._compute_steering_vector_weighted(
                activations_list, 
                rewards,
                device
            )
    
    def _compute_steering_vector_weighted(
        self, 
        activations_list: List[Dict], 
        rewards: List[float],
        device: str = "cuda:0"
    ) -> Dict:
        """
        Compute steering vectors using a reward-weighted approach.
        
        Args:
            activations_list: List of dictionaries containing activations
            rewards: List of corresponding rewards
            device: Device to use for computation
            
        Returns:
            Dictionary of steering vectors for each layer
        """
        # Normalize rewards to weights using softmax
        weights = F.softmax(torch.tensor(rewards, device=device) / 0.7, dim=0)
        
        steering_vectors = {}
        layers_to_use = self.steering_config["layers_to_use"]
        components_to_use = self.steering_config["components_to_use"]
        steering_scale = self.steering_config["steering_scale"]
        
        # Process layer weights
        layer_weights = self.steering_config["layer_weights"]
        if not layer_weights:
            layer_weights = {layer: 1.0 for layer in layers_to_use}
            
        # Process component weights
        component_weights = self.steering_config["component_weights"]
        if not component_weights:
            component_weights = {component: 1.0 for component in components_to_use}
        
        # Calculate steering vectors for each layer and component
        for layer_name in activations_list[0].keys():
            # Parse layer path to get layer index and component
            parts = layer_name.split(".")
            layer_idx = int(parts[2])
            component = ".".join(parts[3:])
            
            # Skip if this layer or component is not in our configured list
            if layer_idx not in layers_to_use or component not in components_to_use:
                continue
                
            # Get layer and component weights
            layer_weight = layer_weights.get(layer_idx, 1.0)
            component_weight = component_weights.get(component, 1.0)
            
            # Stack activations for this layer from all generations
            layer_activations = torch.stack([act[layer_name] for act in activations_list])
            
            # Compute weighted sum based on rewards
            weighted_sum = torch.sum(layer_activations * weights.view(-1, 1, 1, 1), dim=0)
            
            # Compute mean activation
            mean_activation = torch.mean(layer_activations, dim=0)
            
            # Compute steering vector as the difference between weighted sum and mean
            # Apply layer, component, and global scaling factors
            steering_vectors[layer_name] = (weighted_sum - mean_activation) * layer_weight * component_weight * steering_scale
        
        return steering_vectors
    
    def _compute_steering_vector_contrast(
        self, 
        positive_activations: List[Dict], 
        negative_activations: List[Dict],
        device: str = "cuda:0"
    ) -> Dict:
        """
        Compute steering vectors using a contrastive approach.
        
        Args:
            positive_activations: List of activations for positive examples
            negative_activations: List of activations for negative examples
            device: Device to use for computation
            
        Returns:
            Dictionary of steering vectors for each layer
        """
        steering_vectors = {}
        layers_to_use = self.steering_config["layers_to_use"]
        components_to_use = self.steering_config["components_to_use"]
        steering_scale = self.steering_config["steering_scale"]
        
        # Process layer weights
        layer_weights = self.steering_config["layer_weights"]
        if not layer_weights:
            layer_weights = {layer: 1.0 for layer in layers_to_use}
            
        # Process component weights
        component_weights = self.steering_config["component_weights"]
        if not component_weights:
            component_weights = {component: 1.0 for component in components_to_use}
        
        # Calculate steering vectors for each layer and component
        for layer_name in positive_activations[0].keys():
            # Parse layer path to get layer index and component
            parts = layer_name.split(".")
            layer_idx = int(parts[2])
            component = ".".join(parts[3:])
            
            # Skip if this layer or component is not in our configured list
            if layer_idx not in layers_to_use or component not in components_to_use:
                continue
                
            # Get layer and component weights
            layer_weight = layer_weights.get(layer_idx, 1.0)
            component_weight = component_weights.get(component, 1.0)
            
            # Compute mean of positive activations
            positive_mean = torch.mean(torch.stack([act[layer_name] for act in positive_activations]), dim=0)
            
            # Compute mean of negative activations
            negative_mean = torch.mean(torch.stack([act[layer_name] for act in negative_activations]), dim=0)
            
            # Compute steering vector as the difference between positive and negative means
            # Apply layer, component, and global scaling factors
            steering_vectors[layer_name] = (positive_mean - negative_mean) * layer_weight * component_weight * steering_scale
        
        return steering_vectors
    
    def get_steering_metrics(self) -> Dict:
        """
        Get the metrics on steering effectiveness.
        
        Returns:
            Dictionary of metrics on steering effectiveness
        """
        metrics = self.steering_metrics.copy()
        
        # Calculate derived metrics
        if metrics["total_steps"] > 0:
            metrics["steering_usage_rate"] = metrics["steps_with_steering"] / metrics["total_steps"]
        else:
            metrics["steering_usage_rate"] = 0.0
            
        total_selected = metrics["initial_actions_selected"] + metrics["steered_actions_selected"]
        if total_selected > 0:
            metrics["steered_selection_rate"] = metrics["steered_actions_selected"] / total_selected
        else:
            metrics["steered_selection_rate"] = 0.0
            
        if metrics["steps_with_steering"] > 0:
            metrics["steered_improvement_rate"] = metrics["steps_with_steered_improvement"] / metrics["steps_with_steering"]
        else:
            metrics["steered_improvement_rate"] = 0.0
            
        if metrics["initial_reward_avg"]:
            metrics["avg_initial_reward"] = np.mean(metrics["initial_reward_avg"])
        else:
            metrics["avg_initial_reward"] = 0.0
            
        if metrics["steered_reward_avg"]:
            metrics["avg_steered_reward"] = np.mean(metrics["steered_reward_avg"])
        else:
            metrics["avg_steered_reward"] = 0.0
            
        if metrics["avg_reward_improvement"]:
            metrics["mean_reward_improvement"] = np.mean(metrics["avg_reward_improvement"])
        else:
            metrics["mean_reward_improvement"] = 0.0
            
        # Calculate best action source stats
        if metrics["best_action_source"]:
            initial_best = metrics["best_action_source"].count("initial")
            steered_best = metrics["best_action_source"].count("steered")
            total_best = initial_best + steered_best
            
            if total_best > 0:
                metrics["steered_best_rate"] = steered_best / total_best
            else:
                metrics["steered_best_rate"] = 0.0
        else:
            metrics["steered_best_rate"] = 0.0
            
        # Clean up lists that we don't need to return
        del metrics["initial_reward_avg"]
        del metrics["steered_reward_avg"]
        del metrics["avg_reward_improvement"]
        del metrics["best_action_source"]
        
        return metrics
    
    def __call__(self, world: WorldModel, config: SearchConfig) -> BeamSearchResult:
        """
        Run beam search with steering.
        
        Args:
            world: The world model to use for simulation
            config: The search configuration
            
        Returns:
            BeamSearchResult containing the best path and associated information
        """
        # Reset node ID counter
        BeamSearchNode.reset_id()
        
        # Initialize state and root node
        init_state = world.init_state()
        root_node = BeamSearchNode(state=init_state, action=None, reward=0.0)
        
        # Initialize current beam with initial state
        cur_beam = [(root_node, [], 0.0)]  # (node, reward_list, cum_reward)
        terminal_beam = []
        
        # Track activations and steering vectors for each step
        step_activations = {}
        step_rewards = {}
        
        # Reset metrics tracking
        self.steering_metrics = {
            "steps_with_steering": 0,
            "total_steps": 0,
            "initial_actions_selected": 0,
            "steered_actions_selected": 0,
            "initial_reward_avg": [],
            "steered_reward_avg": [],
            "avg_reward_improvement": [],
            "steps_with_steered_improvement": 0,
            "best_action_source": []  # "initial" or "steered" for each step
        }
        
        for depth in range(self.max_depth + 1):
            # Clear activations for this step
            step_activations[depth] = []
            step_rewards[depth] = []
            
            # when depth == max_depth, we need to add the cur_beam to terminal_beam
            new_beam = []
            
            for beam_item in cur_beam:
                node, reward_list, _ = beam_item[:3]
                state = node.state
                
                if self.early_terminate and world.is_terminal(state):
                    terminal_beam.append(beam_item)
                    
                else:
                    if depth == self.max_depth:
                        terminal_beam.append(beam_item)
                        continue
                    
                    # Increment total steps counter
                    self.steering_metrics["total_steps"] += 1
                    
                    # PHASE 1: Generate initial candidate actions without steering
                    logger.debug(f"Depth {depth}: Generating initial actions without steering")
                    config.set_steering_vector(None)  # Ensure no steering for initial generation
                    initial_actions, initial_activations = config.get_actions_with_activations(state)
                    
                    # Process initial actions and compute their rewards
                    initial_action_data = []
                    for action, activation in zip(initial_actions, initial_activations):
                        next_state, aux = world.step(state, action)
                        
                        # Calculate reward
                        fast_reward, fast_reward_aux = config.fast_reward(state, action)
                        reward = config.reward(state, action, **aux, **fast_reward_aux)
                        
                        # Handle tuple reward format
                        if isinstance(reward, tuple):
                            reward, reward_aux = reward
                        
                        # Store action info
                        initial_action_data.append({
                            "action": action,
                            "activation": activation,
                            "next_state": next_state,
                            "reward": reward,
                            "aux": aux,
                            "source": "initial"
                        })
                        
                        # Store activation and reward for steering vector computation
                        step_activations[depth].append(activation)
                        step_rewards[depth].append(reward)
                    
                    # Sort initial actions by reward
                    initial_action_data.sort(key=lambda x: x["reward"], reverse=True)
                    
                    # Calculate average reward for initial actions
                    if initial_action_data:
                        initial_avg_reward = np.mean([data["reward"] for data in initial_action_data])
                        self.steering_metrics["initial_reward_avg"].append(initial_avg_reward)
                    
                    # PHASE 2: Compute steering vector if we have enough initial actions
                    if len(initial_action_data) >= 2:
                        # Track that we're using steering for this step
                        self.steering_metrics["steps_with_steering"] += 1
                        
                        # Take top N candidates for steering computation
                        top_n = min(len(initial_action_data), self.steering_config["num_steering_candidates"])
                        top_activations = [data["activation"] for data in initial_action_data[:top_n]]
                        top_rewards = [data["reward"] for data in initial_action_data[:top_n]]
                        
                        # Compute steering vector
                        logger.debug(f"Depth {depth}: Computing steering vector from {top_n} initial actions")
                        steering_vector = self.compute_steering_vector(top_activations, top_rewards, config.device)
                        
                        # PHASE 3: Generate steered actions using the computed steering vector
                        logger.debug(f"Depth {depth}: Generating steered actions with computed steering")
                        config.set_steering_vector(steering_vector)
                        num_steered = self.steering_config["num_steered_generations"]
                        steered_actions, steered_activations = config.get_actions_with_activations(
                            state, num_actions=num_steered
                        )
                        
                        # Process steered actions and compute their rewards
                        steered_action_data = []
                        for action, activation in zip(steered_actions, steered_activations):
                            next_state, aux = world.step(state, action)
                            
                            # Calculate reward
                            fast_reward, fast_reward_aux = config.fast_reward(state, action)
                            reward = config.reward(state, action, **aux, **fast_reward_aux)
                            
                            # Handle tuple reward format
                            if isinstance(reward, tuple):
                                reward, reward_aux = reward
                            
                            # Store action info
                            steered_action_data.append({
                                "action": action,
                                "activation": activation,
                                "next_state": next_state,
                                "reward": reward,
                                "aux": aux,
                                "source": "steered"
                            })
                        
                        # Sort steered actions by reward
                        steered_action_data.sort(key=lambda x: x["reward"], reverse=True)
                        
                        # Calculate average reward for steered actions and improvement
                        if steered_action_data:
                            steered_avg_reward = np.mean([data["reward"] for data in steered_action_data])
                            self.steering_metrics["steered_reward_avg"].append(steered_avg_reward)
                            
                            # Calculate reward improvement
                            reward_improvement = steered_avg_reward - initial_avg_reward
                            self.steering_metrics["avg_reward_improvement"].append(reward_improvement)
                            
                            # Check if steered actions improved over initial
                            if steered_avg_reward > initial_avg_reward:
                                self.steering_metrics["steps_with_steered_improvement"] += 1
                            
                            # Record best action source (whether best action came from initial or steered)
                            if steered_action_data and initial_action_data:
                                best_initial = max([data["reward"] for data in initial_action_data])
                                best_steered = max([data["reward"] for data in steered_action_data])
                                
                                if best_steered > best_initial:
                                    self.steering_metrics["best_action_source"].append("steered")
                                else:
                                    self.steering_metrics["best_action_source"].append("initial")
                        
                        # PHASE 4: Combine and select best actions from both sets
                        combined_action_data = []
                        
                        if self.steering_config["combine_initial_and_steered"]:
                            # Combine both sets and sort by reward
                            combined_action_data = initial_action_data + steered_action_data
                            combined_action_data.sort(key=lambda x: x["reward"], reverse=True)
                        else:
                            # Use only steered actions if available, otherwise fall back to initial
                            combined_action_data = steered_action_data if steered_action_data else initial_action_data
                        
                        logger.debug(f"Depth {depth}: Combined {len(initial_action_data)} initial and {len(steered_action_data)} steered actions")
                    else:
                        # Not enough initial actions for steering, use only initial actions
                        combined_action_data = initial_action_data
                        logger.debug(f"Depth {depth}: Using only {len(initial_action_data)} initial actions (not enough for steering)")
                    
                    # Create nodes for all selected actions and track source
                    for action_data in combined_action_data:
                        action = action_data["action"]
                        next_state = action_data["next_state"]
                        reward = action_data["reward"]
                        source = action_data["source"]
                        
                        # Add new reward to list of rewards
                        new_reward_list = reward_list + [reward]
                        
                        # Compute new aggregated reward
                        new_reward = self.reward_aggregator(new_reward_list)
                        
                        # Create new node
                        new_node = BeamSearchNode(state=next_state, action=action, reward=reward, parent=node)
                        
                        # Add custom metadata to track source
                        new_node.source = source
                        
                        # Add new node to children of current node
                        node.add_child(new_node)
                        
                        # Track which source was selected
                        if source == "initial":
                            self.steering_metrics["initial_actions_selected"] += 1
                        else:
                            self.steering_metrics["steered_actions_selected"] += 1
                        
                        # Add to new beam
                        if self.sampling_strategy == 'stochastic':
                            # Include action probabilities for stochastic sampling
                            # Use default values since we don't track action probabilities
                            acc_action_prob = 1.0
                            cur_action_prob = 1.0
                            new_beam.append((new_node, new_reward_list, new_reward, (acc_action_prob, cur_action_prob)))
                        else:
                            new_beam.append((new_node, new_reward_list, new_reward))
            
            # Sort new beam by reward
            new_beam.sort(key=lambda x: x[2], reverse=True)
            
            # Sample from new beam
            cur_beam = self._sample(new_beam)
            
            # Reset steering vector for next step
            config.set_steering_vector(None)
            
            # Decay the temperature
            if self.temperature_decay:
                self.temperature *= self.temperature_decay
        
        if not self.early_terminate:
            # add the cur_beam to terminal_beam
            terminal_beam += cur_beam
            
        # Sort terminal beam by reward
        terminal_beam.sort(key=lambda x: x[2], reverse=True)
        
        if self.return_beam:
            # convert terminal_beam to a list of BeamSearchResult
            terminal_beam = [BeamSearchResult(
                                terminal_node=item[0],
                                terminal_state=item[0].state,
                                cum_reward=item[2],
                                tree=root_node,
                                trace=item[0].get_trace(),
                                steering_metrics=self.get_steering_metrics()  # Add steering metrics
                                ) for item in terminal_beam]
            
            return terminal_beam
        
        if len(terminal_beam) == 0:
            return BeamSearchResult(
                terminal_node=None,
                terminal_state=None,
                cum_reward=0,
                tree=root_node,
                trace=[],
                steering_metrics=self.get_steering_metrics()  # Add steering metrics
            )
        
        best_result = terminal_beam[0]
        
        # Add trace analysis for steering effectiveness
        if hasattr(best_result[0], 'get_trace_with_metadata'):
            trace_with_metadata = best_result[0].get_trace_with_metadata()
            # Analyze the trace to see what proportion of steered actions were chosen for the solution
            # (We'll need to implement this method)
        
        result = BeamSearchResult(
            terminal_node=best_result[0],
            terminal_state=best_result[0].state,
            cum_reward=best_result[2],
            tree=root_node,
            trace=best_result[0].get_trace(),
            steering_metrics=self.get_steering_metrics()  # Add steering metrics
        )
        
        return result 