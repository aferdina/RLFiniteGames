import gym

from typing import Optional, Union
from gym import spaces
import numpy as np
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, seed: Optional[int] = None, masking: bool = True):
        self.rng = np.random.default_rng(seed=seed)
        self.masking = masking 
        

    @abstractmethod
    def get_action(self, state: np.array) -> int:
        pass

    @abstractmethod
    def update_masking(self, state: np.array) -> int:
        pass

class FiniteAgent():
    """ The Agent class enables to play different policies for a given evironment

    :param env: The environment to learn from (if registered in Gym, can be str)
    :param use_masking: Whether or not to use invalid action masks during evaluation
    :param seed: Seed for the pseudo random generators
    :param policy_type: the type of the initialization policy
    """

    def __init__(self, env: Union[gym.Env, str] = GridWorld(5), masking: bool = True, seed: Optional[int] = None, policy_type: str = 'uniform') -> None:
        self.env = env
        self.masking = masking
        self.rng = np.random.default_rng(seed=seed)
        # size of action space
        if isinstance(self.env.action_space, spaces.Discrete):
            num_acts = self.env.action_space.n
            self.all_actions = np.arange(num_acts)
        if isinstance(self.env.observation_space, spaces.MultiDiscrete):
            self.obs_shape = self.env.observation_space.nvec
            self.state_type = 'MultiDiscrete'
        if isinstance(self.env.observation_space, spaces.Discrete):
            self.obs_shape = self.env.observation_space.n
            self.state_type = 'Discrete'
        policy_shape = np.append(self.obs_shape, num_acts)

        # Init policy # TODO: Als Funktion auslagern?
        if policy_type == 'uniform':
            self.policy = np.ones(policy_shape)/num_acts
        if policy_type == 'greedy' and self.state_type == 'MultiDiscrete':
            self.policy = np.zeros(policy_shape)
            for i in range(self.obs_shape[0]):
                for j in range(self.obs_shape[1]):
                    k = np.random.randint(num_acts)
                    self.policy[i, j, k] = 1
        if policy_type == 'greedy' and self.state_type == 'Discrete':
            self.policy = np.zeros(policy_shape)
            for i in range(self.obs_shape[0]):
                for j in range(self.obs_shape[1]):
                    k = np.random.randint(num_acts)
                    self.policy[i, j, k] = 1

    def get_action(self, state: Union[np.ndarray,int]) -> int:
        """samples an action of the environments action space for a given state

        Args:
            state (np.ndarray): gamestate of the environment

        Returns:
            action :single action, randomly generated according to policy
        """
        # TODO: assert for state shape
        #assert state.shape == self.obs_shape.shape, "shape of state is {state.shape}, should be {self.obs_shape.shape}"

        # Update Probabilities
        self._update_action_mask_prob(state)
        # Sample Action
        if self.state_type == 'MultiDiscrete':
            state = tuple(state)
        action = self.rng.choice(self.all_actions, p=self.policy[state])
        return action

    def _update_action_mask_prob(self, state: Union[np.ndarray,int]) -> None:
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

