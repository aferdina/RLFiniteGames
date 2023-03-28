""" Implementation of value iteration and policy iteration for finite gym environments
"""
import random
from dataclasses import dataclass
from typing import Union
from enum import Enum
import numpy as np
from gym import Env, spaces
from rlfinitegames.policies.discrete_agents import FiniteAgent
from rlfinitegames.environments.grid_world import GridWorld


# define all possible policy iteration approaches
class PolicyIterationApproaches(Enum):
    """ Enumeration of all possible policy iteration approaches"""
    NAIVE = 'Naive'
    SWEEP = 'Sweep'

# define all possible policy iteration approaches


class MonteCarloApproaches(Enum):
    """ Enumeration of all possible policy iteration approaches"""
    STATE_ACTION_FUNCTION = 'StateActionFunction'
    VALUE_FUNCTION = 'ValueFunction'

# setting required parameters for policy iteration


@dataclass
class MonteCarloPolicyIterationParameters:
    montecarloapproach: MonteCarloApproaches
    stateactionfunctioninit: Union[float, None]
    valuefunctioninit: Union[float, None]
    unvalidstateactionvalue: Union[float, None]


@dataclass
class PolicyIterationParameter:
    """
    Class for Policy Evaluation for Naive or Sweep Approach

    :param approach: String that specifies which approach to use (Naive or Sweep)
    :param epsilon: float variable that determines termination criterium
    :param gamma: float that represents the discount factor
    """
    epsilon: float = 0.01
    gamma: float = 0.95
    approach: PolicyIterationApproaches
    epsilon_greedy: float = 0.1
    epsilon_greedy_decay: float = 1.0


class PolicyIteration():
    """
    Class for Policy Evaluation for Naive or Sweep Approach

    :param policy: define the agent's policy
    :param environment: environment class
    """
    # pylint: disable=line-too-long

    def __init__(self, environment: Union[Env, str] = GridWorld(5), policy=FiniteAgent(), policyparameter: PolicyIterationParameter = PolicyIterationParameter(approach='Naive', epsilon=0.001, gamma=0.95, epsilon_greedy=0.5, epsilon_greedy_decay=0.95), verbose: int = 0, montecarloparameter: MonteCarloPolicyIterationParameters = MonteCarloPolicyIterationParameters(stateactionfunctioninit=20.0, valuefunctioninit=20.0, montecarloapproach="StateActionFunction", unvalidstateacionvalue=-100.0)) -> None:
        # TODO: adding sweep approach to the algorithm
        self.policyparameter = policyparameter  # policy evaluation parameter
        self.environment = environment  # environment class
        self.agent = policy  # agent class
        self.verbose = verbose
        self.montecarloparameter = montecarloparameter
        # Get the number of all possible states depending on Environment Type

        self.state_type = None
        self.init_state_type()

        self.value_func = None
        self.init_value_function()

        self.state_action_function = None
        self.init_state_action_function()
        if isinstance(self.environment.observation_space, spaces.MultiDiscrete):
            if self.montecarloparameter.montecarloapproach == 'StateActionFunction':
                self.state_action_function = np.ones_like(
                    self.environment.observation_space.nvec) * self.montecarloparameter.valuefunctioninit
            elif self.montecarloparameter.montecarloapproach == 'ValueFunction':
                self.value_func = np.ones_like(
                    self.environment.observation_space.nvec) * self.montecarloparameter.valuefunctioninit
            else:
                raise NotImplementedError(f"Unknown montecarlo approach")
        # als Funktion auslagern
        if isinstance(self.environment.observation_space, spaces.Discrete):
            if self.montecarloparameter.montecarloapproach == 'StateActionFunction':
                self.state_action_function = np.ones_like(
                    self.environment.observation_space.n) * self.montecarloparameter.valuefunctioninit
            elif self.montecarloparameter.montecarloapproach == 'ValueFunction':
                self.value_func = np.ones_like(
                    self.environment.observation_space.n) * self.montecarloparameter.valuefunctioninit
            else:
                raise NotImplementedError(f"Unknown montecarlo approach")

        if self.montecarloparameter.montecarloapproach == 'StateActionFunction':
            for state in np.ndindex(self.value_func.shape):
                state_pos_action = self.environment.get_valid_actions(state)
                not_pos_actions = set(self.agent.all_actions) - \
                    set(state_pos_action)
                self.state_action_function[state][list(
                    not_pos_actions)] = self.montecarloparameter.unvalidstateactionvalue

    def init_state_action_function(self)-> None:
        if self.montecarloparameter.montecarloapproach == 'StateActionFunction':
            # Unterscheide zwischen den Faellen
            self.state_action_function = np.ones_like(
                self.agent.policy) * self.montecarloparameter.stateactionfunctioninit
            for state in np.ndindex(self.value_func.shape):
                state_pos_action = self.environment.get_valid_actions(state)
                not_pos_actions = set(self.agent.all_actions) - \
                    set(state_pos_action)
                self.state_action_function[state][list(
                    not_pos_actions)] = self.montecarloparameter.unvalidstateactionvalue

    def init_value_function(self):
        if isinstance(self.environment.observation_space, spaces.MultiDiscrete):
            self.state_action_function = np.ones_like(
                self.environment.observation_space.nvec) * self.montecarloparameter.valuefunctioninit
        elif isinstance(self.environment.observation_space, spaces.Discrete):
            self.state_action_function = np.ones_like(
                self.environment.observation_space.n) * self.montecarloparameter.valuefunctioninit
        else:
            raise NotImplementedError(f"Unknown environment type")

    def init_state_type(self):
        if isinstance(self.environment.observation_space, spaces.MultiDiscrete):
            self.state_type = 'MultiDiscrete'
        elif isinstance(self.environment.observation_space, spaces.Discrete):
            self.state_type = 'Discrete'
        else:
            raise NotImplementedError(f"Unknown environment type")

    def evaluate_state_action_func(self) -> None:
        state_action_func_new = np.zeros_like(
            self.state_action_function.copy())
        for state in np.ndindex(self.value_func.shape):
            state_pos_action = self.environment.get_valid_actions(state)
            not_pos_actions = set(self.agent.all_actions) - \
                set(state_pos_action)
            state_action_func_new[state][list(not_pos_actions)] = -100.0

        number_of_times_played = np.zeros_like(
            self.state_action_function.copy())
        done_converge = False

        while not done_converge:

            # Sample from starting function
            starting_state = self.environment.costum_sample()
            # Create trajectory given the starting state
            action = random.choice(
                self.environment.get_valid_actions(starting_state))
            self.environment.state = starting_state
            trajectory = {"state": [], "action": [], "reward": []}
            done = False
            while not done:
                trajectory["state"].append(self.environment.state)
                trajectory["action"].append(action)
                next_state, reward, done, _ = self.environment.step(action)
                action = self.agent.get_action(next_state)
                trajectory["reward"].append(reward)
            # update estimator for value function, with trick
            print(f"evaluate trajectories")
            for index, (state, action) in reversed(list(enumerate(list(zip(trajectory["state"], trajectory["action"]))))):
                if not any(np.array_equal(arr_state, state) and np.array_equal(arr_action, action) for arr_state, arr_action in list(zip(trajectory['state'], trajectory['action']))[:index]):
                    if self.state_type == 'MultiDiscrete':
                        state_pos = tuple(state)
                    elif self.state_type == 'Discrete':
                        state_pos = state
                    else:
                        raise NotImplementedError(
                            "Only MultiDiscrete and Discrete are supported up to now")
                    discounted_reward = self._calculate_discounted_reward(
                        trajectory["reward"][index:])
                    state_action_func_new[state_pos][action] = 1 / (number_of_times_played[state_pos][action] + 1)*discounted_reward + number_of_times_played[state_pos][action] / (
                        1+number_of_times_played[state_pos][action]) * state_action_func_new[state_pos][action]
                    number_of_times_played[state_pos][action] += 1
            print(
                f"convergence is {np.sum(np.abs(state_action_func_new - self.state_action_function))}")
            if (np.abs(state_action_func_new - self.state_action_function) < self.policyparameter.epsilon).all():
                done_converge = True
                print(f"algo is converged {done_converge}")
                return
            self.state_action_function = state_action_func_new.copy()

    def policy_improvement_state_action_func(self) -> None:
        max_indices = np.argmax(self.state_action_function, axis=-1)
        self.agent.policy = np.ones_like(
            self.agent.policy) * self.policyparameter.epsilon_greedy/(self.environment.action_space.n - 1)

        self.agent.policy[tuple(np.indices(
            self.state_action_function.shape[:-1])) + (max_indices,)] = 1 - self.policyparameter.epsilon_greedy

    def policy_iteration_state_action_function(self) -> None:
        """ using policy iteration to find optimal policy """
        done = False
        counter = 0
        while not done:
            old_policy = self.agent.policy
            self.evaluate_state_action_func()
            self.policy_improvement_state_action_func()
            counter += 1
            print(f"distance {np.sum(np.abs(self.agent.policy - old_policy))}")
            print(f"epsilon greedy is {self.policyparameter.epsilon_greedy}")
            if (np.abs(self.agent.policy - old_policy) < self.policyparameter.epsilon).all():
                done = True
            if counter % 1 == 0:
                self.policyparameter.epsilon_greedy *= self.policyparameter.epsilon_greedy_decay

    def evaluate_value_func(self) -> None:
        """
        use policy evaluation to get the value function from the current policy"""
        # Update the value function according the current policy
        value_func_new = np.zeros_like(self.value_func.copy())
        number_of_times_played = np.zeros_like(self.value_func.copy())
        done_converge = False
        while not done_converge:
            # Sample from starting function
            starting_state = self.environment.observation_space.sample()
            while np.array_equal(starting_state, self.environment.bomb_position) or np.array_equal(starting_state, self.environment.goal_position):
                starting_state = self.environment.observation_space.sample()
            # Create trajectory given the starting state
            self.environment.state = starting_state
            trajectory = {"state": [], "action": [], "reward": []}
            done = False
            while not done:
                action = self.agent.get_action(self.environment.state)
                trajectory["state"].append(self.environment.state)
                trajectory["action"].append(action)
                _next_state, reward, done, _ = self.environment.step(action)
                trajectory["reward"].append(reward)
            # update estimator for value function

            for index, state in reversed(list(enumerate(trajectory["state"]))):
                if not any(np.array_equal(arr, state) for arr in trajectory['state'][:index]):
                    if self.state_type == 'MultiDiscrete':
                        state_pos = tuple(state)
                    elif self.state_type == 'Discrete':
                        state_pos = state
                    else:
                        raise NotImplementedError(
                            "Only MultiDiscrete and Discrete are supported up to now")
                    discounted_reward = self._calculate_discounted_reward(
                        trajectory["reward"][index:])
                    value_func_new[state_pos] = 1 / (number_of_times_played[state_pos] + 1)*discounted_reward + number_of_times_played[state_pos] / (
                        1+number_of_times_played[state_pos]) * value_func_new[state_pos]
                    number_of_times_played[state_pos] += 1

            # check if the new value function is in an epsilon environment of the old value function
            if (np.abs(value_func_new - self.value_func) < self.policyparameter.epsilon).all():
                done_converge = True
                print(f"algo is converged {done_converge}")
                return
            self.value_func = value_func_new.copy()

    def _calculate_discounted_reward(self, reward_trajectory: list[float]) -> float:
        """calculate the discounted reward given a list of reward

        Args:
            reward_trajectory (list[float]): rewards from a trajectory

        Returns:
            float: reward given a trajectory
        """
        discounted_reward = 0.0
        for reward in reversed(reward_trajectory):
            discounted_reward = discounted_reward * self.policyparameter.gamma + reward
        return discounted_reward

    def epsilongreedyimprove(self) -> None:
        """ improve the current policy of the agent by using a policy improvement step
        """
        # improve policy of the agent
        q_values = self._calculate_q_function_general()
        max_indices = np.argmax(q_values, axis=-1)
        self.agent.policy = np.ones_like(
            self.agent.policy) * self.policyparameter.epsilon_greedy/(self.environment.action_space.n - 1)

        self.agent.policy[tuple(np.indices(
            q_values.shape[:-1])) + (max_indices,)] = 1 - self.policyparameter.epsilon_greedy

    def _calculate_q_function_general(self) -> np.ndarray:
        q_values = np.ones_like(self.agent.policy) * - \
            100  # initialize q_values
        # Update weighted Value

        for state in np.ndindex(self.value_func.shape):
            # in discrete case we only need integer values
            if len(state) == 1:
                state = int(state[0])
            # Get and Play all possible actions
            valid_actions = self.environment.get_valid_actions(state)
            for act in valid_actions:
                prob_next_states = self.environment.calculate_probability(
                    state=state, action=act)
                rewards = self.environment.get_rewards(
                    state=state, action=act)

                reward_state_action = 0.0
                value_function_next_step = 0.0
                if self.state_type == 'Discrete':
                    for i, prob_next_state in enumerate(prob_next_states):
                        reward_state_action += prob_next_state * rewards[i]
                    for i, prob_next_state in enumerate(prob_next_states):
                        value_function_next_step += prob_next_state * \
                            self.value_func[i]
                else:
                    reward_state_action = np.sum(prob_next_states * rewards)
                    value_function_next_step = np.sum(
                        prob_next_states * self.value_func)

                q_values[state][act] += reward_state_action + \
                    self.policyparameter.gamma*value_function_next_step
        return q_values

    def policy_iteration(self) -> None:
        """ using policy iteration to find optimal policy """
        done = False
        counter = 0
        while not done:
            old_policy = self.agent.policy
            self.evaluate_value_func()
            self.epsilongreedyimprove()
            counter += 1
            if (np.abs(self.agent.policy - old_policy) < self.policyparameter.epsilon).all():
                done = True
            if counter % 10 == 0:
                self.policyparameter.epsilon_greedy *= self.policyparameter.epsilon_greedy_decay


def main():
    """run policy iteration on grid world game"""
    total_steps = 10
    # set size of grid world
    size = int(input("Please provide size of Grid World Environment: "))
    # initialize agent
    agent = FiniteAgent(env=GridWorld(size=size))
    # initialize grid world environment
    env = GridWorld(size=size)

    # run policy iteration
    algo = PolicyIteration(environment=env, policy=agent, intial_value=20.0)
    # algo.policy_iteration()
    algo.policy_iteration()
    env.reset()

    for _ in range(total_steps):
        action = algo.agent.get_action(env.state)
        _next_state, reward, done, _ = env.step(action)
        print(f"next state: {_next_state}, reward: {reward}, done: {done}")
        print(done)
        if done:
            env.reset()
        env.render()
    print(
        f"Resulting policy: {algo.agent.policy}, Resulting Value Function: {algo.value_func}")
    print(f"Resulting q function: {algo.state_action_function}")


if __name__ == "__main__":
    main()
