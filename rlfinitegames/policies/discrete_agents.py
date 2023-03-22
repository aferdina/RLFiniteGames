""" create agents for finite games"""
from typing import Optional, Union
from gym import Env, spaces
import numpy as np
from rlfinitegames.environments.grid_world import GridWorld
from rlfinitegames.policies.base_agent import BaseAgent

# pylint: disable=too-few-public-methods
class FiniteAgent(BaseAgent):
    """ The Agent class enables to play different policies for a given evironment

    :param env: The environment to learn from (if registered in Gym, can be str)
    :param use_masking: Whether or not to use invalid action masks during evaluation
    :param seed: Seed for the pseudo random generators
    :param policy_type: the type of the initialization policy
    """

    def __init__(self, env: Union[Env, str] = GridWorld(5),
                 masking: bool = True, seed: Optional[int] = None,
                 policy_type: str = 'uniform') -> None:
        super().__init__(seed, masking)
        self.env = env
        # size of action space
        if isinstance(self.env.action_space, spaces.Discrete):
            num_acts = self.env.action_space.n
            self.all_actions = np.arange(num_acts)
        else:
            raise ValueError(
                "The action space is not discrete, and yet not supported")
        if isinstance(self.env.observation_space, spaces.MultiDiscrete):
            self.obs_shape = self.env.observation_space.nvec
            self.state_type = 'MultiDiscrete'
        elif isinstance(self.env.observation_space, spaces.Discrete):
            self.obs_shape = self.env.observation_space.n
            self.state_type = 'Discrete'
        else:
            raise ValueError(
                "The observation space is not discrete or Multidiscrete, and yet not supported")

        policy_shape = np.append(self.obs_shape, num_acts)

        self.policy = None
        self._create_init_policy(policy_type, policy_shape, num_acts)

    def _create_init_policy(self, policy_type: str,
                            policy_shape: np.ndarray,
                            num_acts: int) -> None:
        if policy_type == 'uniform':
            self.policy = np.ones(policy_shape)/num_acts
        elif policy_type == 'greedy' and self.state_type == 'MultiDiscrete':
            self.policy = np.zeros(policy_shape)
            for i in range(self.obs_shape[0]):
                for j in range(self.obs_shape[1]):
                    k = np.random.randint(num_acts)
                    self.policy[i, j, k] = 1
        elif policy_type == 'greedy' and self.state_type == 'Discrete':
            self.policy = np.zeros(policy_shape)
            for i in range(self.obs_shape[0]):
                for j in range(self.obs_shape[1]):
                    k = np.random.randint(num_acts)
                    self.policy[i, j, k] = 1
        else:
            raise ValueError(
                "The policy type is not valid, and yet not supported")

    def get_action(self, state: Union[np.ndarray, int]) -> int:
        """samples an action of the environments action space for a given state

        Args:
            state (np.ndarray): gamestate of the environment

        Returns:
            action :single action, randomly generated according to policy
        """

        # Update Probabilities
        self._update_action_mask_prob(state)
        # Sample Action
        if self.state_type == 'MultiDiscrete':
            state = tuple(state)
        action = self.rng.choice(self.all_actions, p=self.policy[state])
        return action

    def _update_action_mask_prob(self, state: Union[np.ndarray, int]) -> None:
        """Updates the probabilities of all actions for a given state

        Args:
            state (np.ndarray): gamestate of the environment
        """
        if self.masking:
            # Get all possible actions of state:
            pos_actions = self.env.get_valid_actions(state)
            not_pos_actions = set(self.all_actions) - set(pos_actions)
            # Adjust probability of action to legal probabilities
            if self.state_type == 'MultiDiscrete':
                state = tuple(state)
            state_prob = self.policy[state]
            state_prob[list(not_pos_actions)] = 0.0
            state_prob = state_prob/sum(state_prob)
            self.policy[state] = state_prob
