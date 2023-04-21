""" Implementation of value iteration and policy iteration for finite gym environments
"""
import random
from typing import Union
import logging
import numpy as np
from gym import Env, spaces
from rlfinitegames.policies.discrete_agents import FiniteAgent
from itertools import product
from rlfinitegames.logging_module.setup_logger import setup_logger
from rlfinitegames.algorithms.helperclasses import PolicyIterationParameter, MonteCarloPolicyIterationParameters, MonteCarloApproaches


# statics for logging purposes
LOGGINGPATH = "rlfinitegames/logging_module/logfiles/"
FILENAME = "firstvisit"
LOGGING_LEVEL = logging.DEBUG
LOGGING_CONSOLE_LEVEL = logging.INFO
LOGGING_FILE_LEVEL = logging.DEBUG


class MonteCarloPolicyIteration():
    """
    Class for Monte Carlo Policy Evaluation for Naive or Sweep Approach

    :param policy: define the agent's policy
    :param environment: environment class
    """

    def __init__(self, environment: Union[Env, str],
                 policy=FiniteAgent(),
                 policyparameter: PolicyIterationParameter = PolicyIterationParameter(
                     approach='Naive', epsilon=0.001, gamma=0.95, epsilon_greedy=0.5, epsilon_greedy_decay=0.95),
                 verbose: int = 0,
                 montecarloparameter: MonteCarloPolicyIterationParameters = MonteCarloPolicyIterationParameters(stateactionfunctioninit=20.0, valuefunctioninit=20.0, montecarloapproach="StateActionFunction", invalidstateactionvalue=-100.0)) -> None:
        # TODO: adding sweep approach to the algorithm
        self.policyparameter = policyparameter  # policy evaluation parameter
        self.environment = environment  # environment class
        self.agent = policy  # agent class
        self.verbose = verbose
        self.montecarloparameter = montecarloparameter
        # Get the number of all possible states depending on Environment Type

        self.state_type = None
        self.init_state_type()

        self.logger = setup_logger(logger_name=__name__,
                                   logger_level=LOGGING_LEVEL,
                                   log_file=LOGGINGPATH + FILENAME + ".log",
                                   file_handler_level=LOGGING_FILE_LEVEL,
                                   stream_handler_level=LOGGING_CONSOLE_LEVEL,
                                   console_output=True)

        if self.montecarloparameter.montecarloapproach == MonteCarloApproaches.VALUE_FUNCTION.value:
            self.value_func = self.init_value_function(
                self.montecarloparameter.valuefunctioninit)

        if self.montecarloparameter.montecarloapproach == MonteCarloApproaches.STATE_ACTION_FUNCTION.value:
            self.state_action_function = self.init_state_action_function(
                self.montecarloparameter.stateactionfunctioninit)
        else:
            raise NotImplementedError(
                "Other monte carlo approaches are not supported yet")
        
        if not isinstance(self.environment.observation_space, (spaces.MultiDiscrete, spaces.Discrete)):
            raise NotImplementedError("The observation space must be of type Discrete or MultiDiscrete")

    # TODO: adding negativ value for invalid action as parameter
    def init_state_action_function(self, state_action_init: float) -> np.ndarray:
        """ initialize state action function, depending on itialization value

        Args:
            state_action_init (float): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            np.ndarray: _description_
        """
        state_action_function = np.ones_like(
            self.agent.policy) * state_action_init
        # Unterscheide zwischen den Faellen
        if isinstance(self.environment.observation_space, spaces.MultiDiscrete):
            # TODO: not nice implementation yet
            for state in list(product(*[list(range(element)) for element in self.environment.observation_space.nvec.tolist()])):
                state_pos_action = self.environment.get_valid_actions(
                    state)
                not_pos_actions = set(self.agent.all_actions) - \
                    set(state_pos_action)
                state_action_function[state][list(
                    not_pos_actions)] = self.montecarloparameter.invalidstateactionvalue
        elif isinstance(self.environment.observation_space, spaces.Discrete):
            for state in range(self.environment.observation_space.n):
                state_pos_action = self.environment.get_valid_actions(
                    state)
                not_pos_actions = set(self.agent.all_actions) - \
                    set(state_pos_action)
                state_action_function[state][list(
                    not_pos_actions)] = self.montecarloparameter.invalidstateactionvalue
        else:
            raise NotImplementedError(f"Unknown environment type")
        return state_action_function

    def init_value_function(self, value_func_init: float) -> np.ndarray:
        if isinstance(self.environment.observation_space, spaces.MultiDiscrete):
            value_func = np.ones_like(
                self.environment.observation_space.nvec) * value_func_init
        if isinstance(self.environment.observation_space, spaces.Discrete):
            value_func = np.ones_like(
                self.environment.observation_space.n) * value_func_init
        return value_func

    def init_state_type(self):
        if isinstance(self.environment.observation_space, spaces.MultiDiscrete):
            self.state_type = 'MultiDiscrete'
        if isinstance(self.environment.observation_space, spaces.Discrete):
            self.state_type = 'Discrete'

    def evaluate_state_action_func(self) -> None:
        state_action_func_new = self.init_state_action_function(
            state_action_init=0.0)

        number_of_times_played = np.zeros_like(
            self.state_action_function.copy())
        done_converge = False

        while not done_converge:

            # Sample from starting function
            starting_state = self.environment.costum_sample()
            # Create trajectory given the starting state
            action = random.choice(
                self.environment.get_valid_actions(starting_state))
            self.environment.reset()
            self.environment.state = starting_state
            trajectory = {"state": [], "action": [], "reward": []}
            done = False
            while not done:
                trajectory["state"].append(self.environment.state)
                trajectory["action"].append(action)
                next_state, reward, done, _ = self.environment.step(action)
                action = self.agent.get_action(next_state)
                trajectory["reward"].append(reward)
            # TODO: update only for one sample
            # update the estimator for only one sample
            # update estimator for value function, with trick
            self.logger.debug("evaluate trajectories")
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
        """ using epsilon greedy policy improvent to gain improved policy from state action function 
        """
        max_indices = np.argmax(self.state_action_function, axis=-1)
        # TODO: replacing self.agent.policy.copy by shape parameter of policy
        self.agent.policy = np.ones_like(
            self.agent.policy.copy()) * self.policyparameter.epsilon_greedy/(self.environment.action_space.n - 1)

        self.agent.policy[tuple(np.indices(
            self.state_action_function.shape[:-1])) + (max_indices,)] = 1 - self.policyparameter.epsilon_greedy

    def evaluate_value_func(self) -> None:
        """
        use policy evaluation to get the value function from the current policy"""
        # Update the value function according the current policy
        value_func_new = self.init_value_function(value_func_init=0.0)
        number_of_times_played = np.zeros_like(self.value_func.copy())
        done_converge = False
        while not done_converge:
            # Sample from starting function
            starting_state = self.environment.costum_sample()
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
        q_values = np.ones_like(
            self.agent.policy) * self.montecarloparameter.invalidstateactionvalue  # initialize q_values
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

    def policy_iteration_monte_carlo(self) -> None:
        """ using policy iteration with monte carlo samples to find the optimal policy """
        done = False
        counter = 0
        while not done:
            old_policy = self.agent.policy.copy()
            print(
                f"monte carlo approach is {self.montecarloparameter.montecarloapproach}")
            if self.montecarloparameter.montecarloapproach == MonteCarloApproaches.VALUE_FUNCTION.value:
                self.evaluate_value_func()
                self.epsilongreedyimprove()
            elif self.montecarloparameter.montecarloapproach == MonteCarloApproaches.STATE_ACTION_FUNCTION.value:
                self.evaluate_state_action_func()
                self.policy_improvement_state_action_func()
            else:
                raise NotImplementedError("Other methods not implemented yet")

            counter += 1
            if (np.abs(self.agent.policy - old_policy) < self.policyparameter.epsilon).all():
                done = True
            if counter % self.policyparameter.decay_steps == 0:
                self.policyparameter.epsilon_greedy *= self.policyparameter.epsilon_greedy_decay

