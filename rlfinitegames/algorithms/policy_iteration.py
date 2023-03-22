""" Implementation of value iteration and policy iteration for finite gym environments
"""
from dataclasses import dataclass
from typing import Union
from enum import Enum
import numpy as np
import gym
from RL_agent.discreteagent import FiniteAgent
from environments.GridEnv import GridWorld


# define all possible policy iteration approaches
class PolicyIterationApproaches(Enum):
    """ Enumeration of all possible policy iteration approaches"""
    NAIVE = 'Naive'
    SWEEP = 'Sweep'

# setting all requiered policy parameters


@dataclass
class PolicyIterationParameter:
    """
    Class for Policy Evaluation for Naive or Sweep Approach

    :param approach: String that specifies which approach to use (Naive or Sweep)
    :param epsilon: float variable that determines termination criterium
    :param gamma: float that represents the discount factor
    """
    epsilon: float
    gamma: float
    approach: PolicyIterationApproaches


class PolicyIteration():
    """
    Class for Policy Evaluation for Naive or Sweep Approach

    :param policy: define the agent's policy
    :param environment: environment class
    """
    # pylint: disable=line-too-long

    def __init__(self, environment: Union[gym.Env, str] = GridWorld(5), policy=FiniteAgent(), policyparameter: PolicyIterationParameter = PolicyIterationParameter(approach='Naive', epsilon=0.01, gamma=0.9), verbose: int = 0) -> None:
        # TODO: adding second approach to the algorithm
        self.policyparameter = policyparameter  # policy evaluation parameter
        self.environment = environment  # environment class
        self.agent = policy  # agent class
        self.verbose = verbose
        # Get the number of all possible states depending on Environment Type
        if isinstance(self.environment.observation_space, gym.spaces.MultiDiscrete):
            self.value_func = np.zeros(self.environment.observation_space.nvec)
            self.state_type = 'MultiDiscrete'
        if isinstance(self.environment.observation_space, gym.spaces.Discrete):
            self.value_func = np.zeros(self.environment.observation_space.n)
            self.state_type = 'Discrete'

    def evaluate(self) -> None:
        """
        use policy evaluation to get the value function from the current policy"""
        # Update the value function according the current policy
        value_func_new = self.value_func.copy()
        done = False
        while not done:
            # store the new value function in the value_func_new variable
            # TODO: update here
            q_values = self._calculate_q_function_general()
            if self.state_type == "Discrete":
                states = range(self.environment.observation_space.n)
            elif self.state_type == "MultiDiscrete":
                states = [tuple(state) for state in np.ndindex(self.value_func.shape)]
            else:
                raise ValueError(f"Unknown state type: {self.state_type}")
            for state in states:
                new_value = 0.0
                for action in range(self.environment.action_space.n):
                    new_value += self.agent.policy[state][action] * \
                        q_values[state][action]
                value_func_new[state] = new_value
            print(f"New Value Func: {value_func_new}")
            # check if the new value function is in an epsilon environment of the old value function
            if ((value_func_new - self.value_func) < self.policyparameter.epsilon).all():
                done = True
                return value_func_new
            self.value_func = value_func_new.copy()

    def improve(self) -> None:
        """ improve the current policy of the agent by using a policy improvement step
        """
        # improve policy of the agent
        q_values = self._calculate_q_function_general()
        max_indices = np.argmax(q_values, axis=-1)
        self.agent.policy = np.zeros_like(self.agent.policy)
        self.agent.policy[tuple(np.indices(
            q_values.shape[:-1])) + (max_indices,)] = 1

    def _calculate_q_value(self) -> np.ndarray:
        q_values = np.zeros_like(self.agent.policy)  # initialize q_values
        # Update weighted Value
        # TODO: does not working for Discrete Setting
        for state in np.ndindex(self.value_func.shape):  # get all possible states
            # this is enough for deterministic env
            # Get and Play all possible actions
            valid_actions = self.environment.get_valid_actions(state)
            for act in valid_actions:
                # set game state
                self.environment.state = np.array(state)
                # probability of action
                next_state, reward, _, _ = self.environment.step(action=act)
                if weighted:
                    # TODO: Adding transition probability for general case
                    q_values[state][act] += self.agent.policy[state][act] * \
                        (reward + self.policyparameter.gamma *
                         self.value_func[tuple(next_state)])
                else:
                    q_values[state][act] += reward + self.policyparameter.gamma * \
                        self.value_func[tuple(next_state)]
        # TODO: Adding transition probability for general case
        np.sum(q_values, axis=-1, keepdims=True)
        return q_values

    def _calculate_q_function_general(self) -> np.ndarray:
        q_values = np.zeros_like(self.agent.policy)  # initialize q_values
        # Update weighted Value
        # TODO: does not working for Discrete Setting
        for state in np.ndindex(self.value_func.shape):
            # get all possible states
            # this is enough for deterministic env
            # Get and Play all possible actions
            if len(state) == 1:
                state = int(state[0])
            valid_actions = self.environment.get_valid_actions(state)
            for act in valid_actions:
                prob_next_state = self.environment.calculate_probability(
                    state=state, action=act)
                rewards = self.environment.get_rewards(
                    state=state, action=act)
                
                reward_state_action = 0.0
                value_function_next_step = 0.0
                if self.state_type == 'Discrete':
                    for i in range(len(prob_next_state)):
                        reward_state_action += prob_next_state[i] * rewards[i]
                    for i in range(len(prob_next_state)):
                        value_function_next_step += prob_next_state[i] * \
                            self.value_func[i]
                else:
                    reward_state_action = np.sum(prob_next_state * rewards)
                    value_function_next_step = np.sum(prob_next_state * self.value_func)

                q_values[state][act] += reward_state_action + \
                    self.policyparameter.gamma*value_function_next_step
        return q_values

    def policy_iteration(self) -> None:
        """ using policy iteration to find optimal policy """
        done = False
        while not done:
            old_policy = self.agent.policy
            self.evaluate()
            self.improve()
            if ((self.agent.policy - old_policy) < self.policyparameter.epsilon).all():
                done = True


if __name__ == "__main__":
    size = int(input("Please provide size of Grid World Environment: "))
    agent = FiniteAgent(env=GridWorld(size=size))
    env = GridWorld(size=size)
    algo = PolicyIteration(environment=env, policy=agent)
    algo.policy_iteration()
    env.reset()
    for _ in range(20):
        action = algo.agent.get_action(env.state)
        next_state, reward, done, _ = env.step(action)
        if done:
            env.reset()
        env.render()
    print(
        f"Resulting policy: {algo.agent.policy}, Resulting Value Function: {algo.value_func}")
