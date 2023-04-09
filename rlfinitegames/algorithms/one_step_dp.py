""" Implementation of value iteration and policy iteration for finite gym environments
"""
import random
from dataclasses import dataclass
from typing import Union
import numpy as np
from gym import Env, spaces
from rlfinitegames.policies.discrete_agents import FiniteAgent
from rlfinitegames.environments.grid_world import GridWorld
from rlfinitegames.environments.ice_vendor import IceVendor, GameConfig
from itertools import product
import logging
from rlfinitegames.logging_module.setup_logger import setup_logger

# statics for logging purposes
LOGGINGPATH = "rlfinitegames/logging_module/logfiles/"
FILENAME = "one_step_dp"
LOGGING_LEVEL = logging.INFO
LOGGING_CONSOLE_LEVEL = logging.INFO
LOGGING_FILE_LEVEL = logging.INFO
# initialize logging module


@dataclass
class OneStepDynamicProgrammingInitConfig:
    stateactionfunctioninit: Union[float, None] = None
    invalidstateactionvalue: Union[float, None] = None


@dataclass
class OneStepDynamicProgrammingParameters:
    """
    Class for Policy Evaluation for Naive or Sweep Approach

    :param approach: String that specifies which approach to use (Naive or Sweep)
    :param epsilon: float variable that determines termination criterium
    :param gamma: float that represents the discount factor
    """
    epsilon: float = 0.01
    gamma: float = 0.95
    epsilon_greedy: float = 0.1
    epsilon_greedy_decay: float = 1.0
    decay_steps: int = 1


class OneStepDynamicProgramming():
    """
    Class for Policy Evaluation for Naive or Sweep Approach

    :param policy: define the agent's policy
    :param environment: environment class
    """
    # pylint: disable=line-too-long

    def __init__(self, environment: Union[Env, str] = GridWorld(5), policy=FiniteAgent(), policyparameter: OneStepDynamicProgrammingParameters = OneStepDynamicProgrammingParameters(), verbose: int = 0, init_parameter: OneStepDynamicProgrammingInitConfig = OneStepDynamicProgrammingInitConfig(stateactionfunctioninit=20.0, invalidstateactionvalue=-1000000)) -> None:
        # TODO: adding sweep approach to the algorithm
        self.policyparameter = policyparameter  # policy evaluation parameter
        self.environment = environment  # environment class
        self.agent = policy  # agent class
        self.verbose = verbose
        self.init_parameter = init_parameter
        # Get the number of all possible states depending on Environment Type

        self.state_type = None
        self.init_state_type()

        self.state_action_function = self.init_state_action_function(
            self.init_parameter.stateactionfunctioninit)

        self.logger = setup_logger(logger_name=__name__,
                                   logger_level=LOGGING_LEVEL,
                                   log_file=LOGGINGPATH + FILENAME + ".log",
                                   file_handler_level=LOGGING_FILE_LEVEL,
                                   stream_handler_level=LOGGING_CONSOLE_LEVEL,
                                   console_output=True)

    def init_state_action_function(self, state_action_init: float) -> np.ndarray:
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
                    not_pos_actions)] = self.init_parameter.invalidstateactionvalue
        elif isinstance(self.environment.observation_space, spaces.Discrete):
            for state in range(self.environment.observation_space.n):
                state_pos_action = self.environment.get_valid_actions(
                    state)
                not_pos_actions = set(self.agent.all_actions) - \
                    set(state_pos_action)
                state_action_function[state][list(
                    not_pos_actions)] = self.init_parameter.invalidstateactionvalue
        else:
            raise NotImplementedError(f"Unknown environment type")
        return state_action_function

    def init_state_type(self) -> None:
        """ initializes the state type of the gym environment

        Raises:
            NotImplementedError: Up to now only `Discrete` and `Multidiscrete` gym observation spaces are supported
        """
        if isinstance(self.environment.observation_space, spaces.MultiDiscrete):
            self.state_type = 'MultiDiscrete'
        elif isinstance(self.environment.observation_space, spaces.Discrete):
            self.state_type = 'Discrete'
        else:
            # otherwise not implemented yet
            try:
                raise NotImplementedError("Unknown environment type")
            except NotImplementedError as e:
                self.logger.exception(str(e))

    def sarsa_on_policy_evaluation(self) -> None:
        counter = 0
        state_action_func_new = self.init_state_action_function(
            state_action_init=0.0)

        number_of_times_played = np.zeros_like(
            self.state_action_function.copy())
        done_converge = False

        while not done_converge:

            # Sample from starting function
            state = self.environment.costum_sample()
            # Create trajectory given the starting state
            action = random.choice(self.environment.get_valid_actions(state))
            self.environment.reset()
            self.environment.state = state
            done = False
            while not done:
                # sample reward, next state, done
                next_state, reward, done, _ = self.environment.step(action)
                next_action = self.agent.get_action(next_state)
                alpha = self.get_alpha(
                    state=state, action=action, times_played=number_of_times_played)
                if self.state_type == 'MultiDiscrete':
                    state_pos = tuple(state)
                    next_state_pos = tuple(next_state)
                elif self.state_type == 'Discrete':
                    state_pos = state
                    next_state_pos = next_state
                else:
                    raise NotImplementedError(
                        "Only MultiDiscrete and Discrete are supported up to now")
                # updating state action function
                if done:
                    state_action_func_new[state_pos][action] = state_action_func_new[state_pos][action] + alpha * (
                        reward - state_action_func_new[state_pos][action])
                else:
                    # Not good for the case when the terminal state is not really a terminal state
                    state_action_func_new[state_pos][action] = state_action_func_new[state_pos][action] + alpha * (
                        reward + self.policyparameter.gamma * self.state_action_function[next_state_pos][next_action] - state_action_func_new[state_pos][action])
                number_of_times_played[state_pos][action] += 1
                action = next_action
                state = next_state
                self.get_epsilon_greedy_improved_policy(
                    state_action_function=state_action_func_new)
                counter += 1
                if counter % 1000 == 0:
                    self.policyparameter.epsilon_greedy *= self.policyparameter.epsilon_greedy_decay
                    self.logger.debug(
                        f"epsilon_greedy is {self.policyparameter.epsilon_greedy}")
            self.logger.debug(
                f"convergence is {np.sum(np.abs(state_action_func_new - self.state_action_function))}")
            if (np.abs(state_action_func_new - self.state_action_function) < self.policyparameter.epsilon).all():
                done_converge = True
                self.logger.info(f"algo is converged {done_converge}")
                return
            self.state_action_function = state_action_func_new.copy()

    def sarsa_evaluate_state_action_func(self) -> None:
        state_action_func_new = self.init_state_action_function(
            state_action_init=0.0)

        number_of_times_played = np.zeros_like(
            self.state_action_function.copy())
        done_converge = False

        while not done_converge:

            # Sample from starting function
            state = self.environment.costum_sample()
            # Create trajectory given the starting state
            action = self.agent.get_action(state)
            self.environment.reset()
            self.environment.state = state
            done = False
            while not done:
                # sample reward, next state, done
                next_state, reward, done, _ = self.environment.step(action)
                next_action = self.agent.get_action(next_state)
                alpha = self.get_alpha(
                    state=state, action=action, times_played=number_of_times_played)
                if self.state_type == 'MultiDiscrete':
                    state_pos = tuple(state)
                    next_state_pos = tuple(next_state)
                elif self.state_type == 'Discrete':
                    state_pos = state
                    next_state_pos = next_state
                else:
                    raise NotImplementedError(
                        "Only MultiDiscrete and Discrete are supported up to now")
                # updating state action function
                if done:
                    state_action_func_new[state_pos][action] = state_action_func_new[state_pos][action] + alpha * (
                        reward - state_action_func_new[state_pos][action])
                else:
                    state_action_func_new[state_pos][action] = state_action_func_new[state_pos][action] + alpha * (
                        reward + self.policyparameter.gamma * self.state_action_function[next_state_pos][next_action] - state_action_func_new[state_pos][action])
                number_of_times_played[state_pos][action] += 1
                action = next_action
                state = next_state
            print(
                f"convergence is {np.sum(np.abs(state_action_func_new - self.state_action_function))}")
            if (np.abs(state_action_func_new - self.state_action_function) < self.policyparameter.epsilon).all():
                done_converge = True
                print(f"algo is converged {done_converge}")
                return
            self.state_action_function = state_action_func_new.copy()

    def get_alpha(self, state: Union[int, np.ndarray], action: int, times_played: np.ndarray, rate: float = 1.0) -> float:
        if self.state_type == 'MultiDiscrete':
            state_pos = tuple(state)
        elif self.state_type == 'Discrete':
            state_pos = state
        else:
            raise NotImplementedError(
                "Only MultiDiscrete and Discrete are supported up to now")
        return 1 / (1 + times_played[state_pos][action]) ** rate

    def q_learning_off_policy(self, policy) -> None:
        """ implementation from algorithm q_learning_off_policy

        Args:
            policy (_type_): _description_
        """
        self.logger.info("q_learning_off_policy is starting")
        self.q_learning_off_policy_td_control(behavior_policy=policy)
        self.get_greedy_policy(state_action_function=None)

    def get_epsilon_greedy_improved_policy(self, state_action_function: Union[np.ndarray, None]) -> None:
        self.logger.debug(f"{self.state_action_function}")
        if state_action_function is None:
            max_indices = np.argmax(self.state_action_function, axis=-1)
            self.agent.policy = np.ones_like(
                self.agent.policy) * self.policyparameter.epsilon_greedy/(self.environment.action_space.n - 1)
            self.agent.policy[tuple(np.indices(
                self.state_action_function.shape[:-1])) + (max_indices,)] = 1 - self.policyparameter.epsilon_greedy
        else:
            max_indices = np.argmax(state_action_function, axis=-1)
            self.agent.policy = np.ones_like(
                self.agent.policy) * self.policyparameter.epsilon_greedy/(self.environment.action_space.n - 1)

            self.agent.policy[tuple(np.indices(
                state_action_function.shape[:-1])) + (max_indices,)] = 1 - self.policyparameter.epsilon_greedy

    def get_greedy_policy(self, state_action_function: Union[np.ndarray, None]) -> None:
        self.logger.debug(f"{self.state_action_function}")
        if state_action_function is None:
            max_indices = np.argmax(self.state_action_function, axis=-1)
        else:
            max_indices = np.argmax(state_action_function, axis=-1)
        self.agent.policy = np.zeros_like(
            self.agent.policy)

        self.agent.policy[tuple(np.indices(
            self.state_action_function.shape[:-1])) + (max_indices,)] = 1

    def q_learning_off_policy_td_control(self, behavior_policy: np.ndarray) -> None:
        self.logger.debug(f"starting policy is given by {behavior_policy}")
        self.agent.policy = behavior_policy.copy()
        number_of_times_played = np.zeros_like(
            self.state_action_function.copy())

        self.state_action_function = self.init_state_action_function(
            state_action_init=10.0)
        state_action_func_new = self.state_action_function.copy()
        self.logger.debug(
            f"starting state action function is given by {state_action_func_new}")

        done_converge = False

        while not done_converge:
            # Sample from starting function
            state = np.array([0, 0], dtype=np.int32)
            # Create trajectory given the starting state
            self.environment.reset()
            self.environment.state = state
            done = False
            while not done:
                # sample reward, next state, done
                action = self.agent.get_action(state)
                self.logger.debug(f"action is {action}")
                self.logger.debug(f"state is {self.environment.state}")
                next_state, reward, done, _ = self.environment.step(action)
                alpha = self.get_alpha(
                    state=state, action=action, times_played=number_of_times_played.copy())
                self.logger.debug(f"alpha is {alpha}")
                self.logger.debug(f"reward is given by {reward}")
                if self.state_type == 'MultiDiscrete':
                    state_pos = tuple(state)
                    next_state_pos = tuple(next_state)
                elif self.state_type == 'Discrete':
                    state_pos = state
                    next_state_pos = next_state
                else:
                    raise NotImplementedError(
                        "Only MultiDiscrete and Discrete are supported up to now")
                # updating state action function
                if done:
                    state_action_func_new[state_pos][action] = state_action_func_new[state_pos][action] + alpha * (
                        reward - state_action_func_new[state_pos][action])
                else:
                    state_action_func_new[state_pos][action] = state_action_func_new[state_pos][action] + alpha * (
                        reward + self.policyparameter.gamma * np.max(state_action_func_new[next_state_pos]) - state_action_func_new[state_pos][action])
                number_of_times_played[state_pos][action] += 1
                state = next_state
                self.logger.debug(
                    f"state action value is {state_action_func_new[state_pos][action]}")
                self.logger.debug(f"state was {state}, action was {action}")
                self.logger.debug(50*"*")
                self.logger.debug(
                    f"state action function is given by {state_action_func_new}")
            self.logger.debug(
                f"convergence is {np.sum(np.abs(state_action_func_new - self.state_action_function))}")
            if (np.abs(state_action_func_new - self.state_action_function) < self.policyparameter.epsilon).all():
                done_converge = True
                self.logger.info(
                    f"q_learning_off_policy_td_control is converged {done_converge}")
                return
            self.state_action_function = state_action_func_new.copy()

    def policy_improvement_state_action_func(self) -> None:
        max_indices = np.argmax(self.state_action_function, axis=-1)
        self.agent.policy = np.ones_like(
            self.agent.policy) * self.policyparameter.epsilon_greedy/(self.environment.action_space.n - 1)

        self.agent.policy[tuple(np.indices(
            self.state_action_function.shape[:-1])) + (max_indices,)] = 1 - self.policyparameter.epsilon_greedy
