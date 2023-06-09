""" implementation of grid search environment from lecture notes"""
from typing import Optional, Tuple
from gym import spaces, Env
import numpy as np
import matplotlib.pyplot as plt

START_STATE = np.array([0, 0], dtype=np.int32)
# pylint: disable=too-many-instance-attributes


class GridWorld(Env):
    """ grid world environment """

    def __init__(self, size: int = 5) -> None:
        """ init grid world environment

        Args:
            size (int): size of the grid world environment
        """
        assert isinstance(
            size, int), f"size has to be an int but is {type(size)}"
        assert size > 2, f"size has to be greater than 2 but is {size}"
        self.size = size
        self.observation_space = spaces.MultiDiscrete(
            [size, size], dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.action_to_direction = {
            0: np.array([1, 0]),        # going down
            1: np.array([0, 1]),        # going right
            2: np.array([-1, 0]),       # going up
            3: np.array([0, -1]),       # going left
        }
        self.goal_position = np.array(
            [size-1, size-1], dtype=np.int32)   # position of the goal
        self.bomb_position = np.array(
            [size-2, size-2], dtype=np.int32)   # position of the bomb

        # reset environment
        self.state = None
        self.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        valid_action_space = self.get_valid_actions(self.state)

        done = False
        if action not in valid_action_space:
            return self.state, 0.0, done, {}

        next_state = self.state + self.action_to_direction[action]
        if np.array_equal(next_state, self.goal_position):
            done = True
            reward = 10.0
        elif np.array_equal(next_state, self.bomb_position):
            done = True
            reward = -10.0
        else:
            reward = -1.0

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

    def get_valid_actions(self, agent_position: np.ndarray) -> list[int]:
        """ get a list with all valid actions given a specific position

        Args:
            agent_position (np.ndarray): position of the agent in the grid

        Returns:
            list[int]: list with all valid actions for the specific position
        """
        valid_actions = [0, 1, 2, 3]
        if agent_position[0] == 0:
            valid_actions.remove(2)
        if agent_position[1] == 0:
            valid_actions.remove(3)
        if agent_position[0] == self.size-1:
            valid_actions.remove(0)
        if agent_position[1] == self.size-1:
            valid_actions.remove(1)
        return valid_actions

    def calculate_probability(self, state: np.ndarray, action: int) -> list[float]:
        """calculate the probability of transitioning to next states given state and action"""
        prob_next_state = np.zeros(self.observation_space.nvec)
        prob_next_state[tuple(state+self.action_to_direction[action])] = 1.0
        return prob_next_state

    def render(self) -> None:
        """ render the current state to the screen
        """

        # translate positions of agent, bomb and target in a matrix
        grid = np.zeros((self.size, self.size))
        grid[self.state[0]][self.state[1]] = 1
        grid[self.goal_position[0]][self.goal_position[1]] = 2
        grid[self.bomb_position[0]][self.bomb_position[1]] = 3

        # create a heatmap from the data
        plt.figure(figsize=(self.size-2, self.size-2))
        plt.imshow(grid, cmap='gray', interpolation='none')
        axis = plt.gca()
        axis.set_xticks(np.arange(self.size)-0.5, labels=np.arange(self.size))
        axis.set_yticks(np.arange(self.size)-0.5, labels=np.arange(self.size))
        plt.grid(color='b', lw=2, ls='-')

        # plot positions of the agent, the bomb and the target position
        plt.text(self.state[1], self.state[0], "A", color="lime", size=12,
                 verticalalignment='center', horizontalalignment='center', fontweight='bold')
        plt.text(self.goal_position[1], self.goal_position[0], "T", color="lime", size=12,
                 verticalalignment='center', horizontalalignment='center', fontweight='bold')
        plt.text(self.bomb_position[1], self.bomb_position[0], "B", color="lime", size=12,
                 verticalalignment='center', horizontalalignment='center', fontweight='bold')

        # show the plot
        plt.show()

    # TODO: how to handle the situation of unused variables for general classes
    # maybe right kwargs in base class
    def get_rewards(self, state: np.ndarray, action: int) -> list[float]:
        """get the reward for the next state given the current state and the action

        Args:
            state (np.ndarray): state to retrieve the rewards from
            action (int): action to retrieve the rewards from

        Returns:
            list[float]: list with next rewards 
        """
        rewards = np.ones(self.observation_space.nvec)*-1
        rewards[tuple(self.goal_position)] = 10
        rewards[tuple(self.bomb_position)] = -10
        return rewards

    def costum_sample(self) -> np.ndarray:
        """sample a random state from the environment

        Returns:
            np.ndarray: random starting state from the environment for algorithm
        """
        # sample = self.observation_space.sample()
        # while np.array_equal(sample,
        #                     self.bomb_position) or np.array_equal(sample,
        #                                                           self.goal_position):
        #     sample = self.observation_space.sample()
        sample = START_STATE
        return sample

    def get_terminal_states(self) -> list[np.ndarray]:
        """ get list of all terminal states from the environment

        Returns:
            list[np.ndarray]: terminal states from the environment
        """
        return [self.goal_position, self.bomb_position]
