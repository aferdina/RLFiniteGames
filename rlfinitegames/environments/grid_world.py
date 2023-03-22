from gym import spaces, Env
import numpy as np
import matplotlib.pyplot as plt


class GridWorld(Env):
    def __init__(self, size):
        assert isinstance(
            size, int), f"size has to be an int but is {type(size)}"
        assert size > 2, f"size has to be greater than 2 but is {size}"
        self.size = size
        self.observation_space = spaces.MultiDiscrete([size, size])
        self.action_space = spaces.Discrete(4)
        self.action_to_direction = {
            0: np.array([1, 0]),        # going down
            1: np.array([0, 1]),        # going right
            2: np.array([-1, 0]),       # going up
            3: np.array([0, -1]),       # going left
        }
        self.goal_position = [size-1, size-1]   # position of the goal
        self.bomb_position = [size-2, size-2]   # position of the bomb
        self.state = None
        self.valid_action_space = None

        # needed for build_transition_prob
        self.all_states = np.prod(self.observation_space.nvec)
        self.lookup_dict = {count: state for count, state in zip(
            range(self.all_states), np.ndindex(np.zeros(self.observation_space.nvec).shape))}
        self.lookup_dict_rev = {y: x for x, y in self.lookup_dict.items()}
        self.transition_probs = np.zeros(
            (self.action_space.n, self.all_states, self.all_states))
        self._build_transition_probabilities()

        # reset environment
        self.reset()

    def step(self, action):
        valid_action_space = self.get_valid_actions(self.state)

        done = False
        if action not in valid_action_space:
            return self.state, 0, done, {}

        next_state = self.state + self.action_to_direction[action]
        if np.array_equal(self.state, self.goal_position):
            done = True
            reward = 10
        elif np.array_equal(self.state, self.bomb_position):
            done = True
            reward = -10
        else:
            reward = -1

        self.state = next_state
        return next_state, reward, done, {}

    def reset(self):
        self.state = np.array([0, 0], dtype=np.int32)
        self.valid_action_space = [0, 1]

    def get_valid_actions(self, agent_position):
        """ get a list with all valid actions given a specific position

        Args:
            agent_position (): _description_

        Returns:
            _type_: _description_
        """
        if agent_position[0] == 0:
            if agent_position[1] == 0:
                return [0, 1]
            elif agent_position[1] == self.size-1:
                return [0, 3]
            else:
                return [0, 1, 3]
        elif agent_position[0] == self.size-1:
            if agent_position[1] == 0:
                return [1, 2]
            elif agent_position[1] == self.size-1:
                return [2, 3]
            else:
                return [1, 2, 3]
        else:
            if agent_position[1] == 0:
                return [0, 1, 2]
            elif agent_position[1] == self.size-1:
                return [0, 2, 3]
            else:
                return [0, 1, 2, 3]

    def calculate_probability(self, state: np.ndarray, action: int):
        prob_next_state = np.zeros(self.observation_space.nvec)
        prob_next_state[tuple(state+self.action_to_direction[action])] = 1
        return prob_next_state

    def _build_transition_probabilities(self):
        for act in range(self.action_space.n):
            for count in range(self.all_states):
                # agent position
                position = self.lookup_dict[count]
                self.state = position
                valid_action = self.get_valid_actions(position)
                if act in valid_action:
                    self.step(action=act)
                    position_after = tuple(self.state)
                    count_after = self.lookup_dict_rev[position_after]
                    self.transition_probs[act, count, count_after] = 1
        # Reset agent position
        self.reset()

    def render(self):

        # translate positions of agent, bomb and target in a matrix
        grid = np.zeros((self.size, self.size))
        grid[self.state[0]][self.state[1]] = 1
        grid[self.goal_position[0]][self.goal_position[1]] = 2
        grid[self.bomb_position[0]][self.bomb_position[1]] = 3

        # create a heatmap from the data
        plt.figure(figsize=(self.size-2, self.size-2))
        plt.imshow(grid, cmap='gray', interpolation='none')
        ax = plt.gca()
        ax.set_xticks(np.arange(self.size)-0.5, labels=np.arange(self.size))
        ax.set_yticks(np.arange(self.size)-0.5, labels=np.arange(self.size))
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

    def get_rewards(self, state: np.ndarray, action: int):
        rewards = np.ones(self.observation_space.nvec)*-1
        rewards[tuple(self.goal_position)] = 10
        rewards[tuple(self.bomb_position)] = -10
        return rewards


def main():
    game = GridWorld(size=5)
    game.reset()
    for _ in range(10):
        valid_actions = game.get_valid_actions(game.state)
        action = np.random.choice(valid_actions)
        prob_next_state = game.calculate_probability(game.state, action)
        print(action, game.state)
        print(prob_next_state)
        next_state, reward, done, _info = game.step(action)


if __name__ == "__main__":
    main()
