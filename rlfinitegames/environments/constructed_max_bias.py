""" implementation of grid search environment from lecture notes"""
from typing import Optional, Tuple
from gym import spaces, Env
import numpy as np
import matplotlib.pyplot as plt

START_STATE = 0
# pylint: disable=too-many-instance-attributes


class ConstructedMaxBias(Env):
    """ grid world environment """

    def __init__(self, number_arms: int) -> None:
        """ example from sutton 6.7

        Args:
            number_arms (int): number of possible actions in state `D` aka `1`
        """
        assert isinstance(
            number_arms, int), f"number_arms has to be an int but is {type(number_arms)}"
        assert number_arms > 2, f"number_arms has to be greater than 2 but is {number_arms}"
        self.number_arms = number_arms
        self.observation_space = spaces.Discrete(3)
        # the state `0` is equal to `C` and the state `1` is equal to `D` and `2` is equal to terminal state
        self.action_space = spaces.Discrete(self.number_arms)

        # reset environment
        self.state = None
        self.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        valid_action_space = self.get_valid_actions(self.state)

        done = False
        if action not in valid_action_space:
            return self.state, 0, done, {}

        next_state = self.state + self.action_to_direction[action]
        if np.array_equal(next_state, self.goal_position):
            done = True
            reward = 10
        elif np.array_equal(next_state, self.bomb_position):
            done = True
            reward = -10
        else:
            reward = -1

        self.state = next_state
        return next_state, reward, done, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> None:
        super().reset(seed=seed)
        self.state = START_STATE

    def get_valid_actions(self, agent_position: int) -> list[int]:
        """ get valid actions for a given agent position

        Args:
            agent_position (int): agent position to get the valid actions from

        Raises:
            ValueError: Error is raised if the given position is not valid

        Returns:
            list[int]: list of all possible actions
        """
        if agent_position == 0:
            return [0, 1,]
        elif agent_position == 1:
            return list(range(self.action_space.n))
        else:
            raise ValueError("agent_position is not defined")

    def calculate_probability(self, state: int, action: int) -> np.ndarray:
        """calculate the probability of transitioning to next states given state and action"""
        prob_next_state = np.zeros(self.observation_space.n)
        if state == 0:
            if action == 0:
                prob_next_state[1] = 1.0
            else:
                prob_next_state[2] = 1.0
        elif state == 1:
            prob_next_state[2] = 1.0
        elif state == 2:
            prob_next_state[2] = 1.0
        else:
            raise ValueError("state is not defined")
        return prob_next_state

    def render(self):
        pass

    # TODO: how to handle the situation of unused variables for general classes
    # maybe right kwargs in base class
    def get_rewards(self, state: np.ndarray, action: int) -> float:
        """get the reward for the next state given the current state and the action"""
        rewards = np.zeros(self.observation_space.n)
        if state == 0 or state == 2:
            return rewards
        elif state == 1:
            rewards[2] = np.random.normal(loc=-0.1, scale=1.0)
            return rewards
        else:
            raise ValueError(f"state {state} is not in observation space")

    def costum_sample(self) -> np.ndarray:
        """sample a random state from the environment"""
        # sample = self.observation_space.sample()
        # while np.array_equal(sample,
        #                     self.bomb_position) or np.array_equal(sample,
        #                                                           self.goal_position):
        #     sample = self.observation_space.sample()
        sample = START_STATE
        return sample
