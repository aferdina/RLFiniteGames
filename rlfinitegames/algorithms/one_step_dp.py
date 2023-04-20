""" Implementation of value iteration and policy iteration for finite gym environments
"""
import random
from dataclasses import dataclass
from itertools import product
from enum import Enum
import logging
from typing import Union
import numpy as np
from gym import Env, spaces
from scipy.stats import entropy
from rlfinitegames.policies.discrete_agents import FiniteAgent
from rlfinitegames.environments.grid_world import GridWorld
from rlfinitegames.logging_module.setup_logger import setup_logger

# statics for logging purposes
LOGGINGPATH = "rlfinitegames/logging_module/logfiles/"
FILENAME = "one_step_dp"
LOGGING_LEVEL = logging.DEBUG
LOGGING_CONSOLE_LEVEL = logging.INFO
LOGGING_FILE_LEVEL = logging.DEBUG
# initialize logging module

# TODO: not use lazy % formatting in logging function
# pylint: disable=W1203


@dataclass
class RunTimeMethod(Enum):
    """ Dataclass to specify the method to run the algorithm. If episodes is used, then 
    the algorithm is running until a number of episodes have been completed. Otherwise,
    the algorithm
    is running until a convergence criterion has been reached.
    """
    EPISODES = "episodes"
    CRITERION = "criterion"


@dataclass
class OneStepDynamicProgrammingInitConfig:
    """ Dataclass to specify the initialization values for the stateaction function in the algorithm
    """
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
    decay_steps: int = 5
    run_time_method: RunTimeMethod = "episodes"
    episodes: Union[int, None] = None
    rate: float = 1.0


class OneStepDynamicProgramming():
    """
    Class for Policy Evaluation for Naive or Sweep Approach

    :param policy: define the agent's policy
    :param environment: environment class
    """
    # pylint: disable=line-too-long

    def __init__(self, environment: Union[Env, str] = GridWorld(5), policy=FiniteAgent(), policyparameter: OneStepDynamicProgrammingParameters = OneStepDynamicProgrammingParameters(), verbose: int = 0, init_parameter: OneStepDynamicProgrammingInitConfig = OneStepDynamicProgrammingInitConfig(stateactionfunctioninit=20.0, invalidstateactionvalue=-1000000)) -> None:
        self.policyparameter = policyparameter  # policy evaluation parameter
        self.environment = environment  # environment class
        self.agent = policy  # agent class
        self.verbose = verbose
        self.init_parameter = init_parameter
        # Get the number of all possible states depending on Environment Type

        self.state_type = None
        self._init_state_type()

        self.state_action_function = self._init_state_action_function(
            self.init_parameter.stateactionfunctioninit)

        self.logger = setup_logger(logger_name=__name__,
                                   logger_level=LOGGING_LEVEL,
                                   log_file=LOGGINGPATH + FILENAME + ".log",
                                   file_handler_level=LOGGING_FILE_LEVEL,
                                   stream_handler_level=LOGGING_CONSOLE_LEVEL,
                                   console_output=True)

    def sarsa_on_policy_control_terminating(self) -> None:
        """ sarsa on policy control for terminmatic mdps from lecture, algo 24

        Raises:
            NotImplementedError: Error if observation space is not `MultiDiscrete` or `Discrete`
        """
        counter = 0
        self.logger.info("Sarsa on policy evaluation starting")
        state_action_func_new = self._init_state_action_function(
            state_action_init=0.0)

        number_of_times_played = np.zeros_like(
            self.state_action_function.copy())
        done_converge = False

        while not done_converge:
            self.logger.debug("Creating a new trajectory")
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
                self.logger.debug(
                    f"next_state is {next_state}, reward is {reward} and done is {done}")
                next_action = self.agent.get_action(next_state)
                self.logger.debug(f"next action is {action}")
                alpha = self._get_alpha(
                    state=state, action=action, times_played=number_of_times_played, rate=self.policyparameter.rate)
                self.logger.debug(f"alpha is given by {alpha}")
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
                self._get_epsilon_greedy_improved_policy(
                    state_action_function=state_action_func_new.copy())
            self.logger.info(
                f"convergence is {np.sum(np.abs(state_action_func_new - self.state_action_function))}")
            if (np.abs(state_action_func_new - self.state_action_function) < self.policyparameter.epsilon).all():
                done_converge = True
                self.logger.info(f"algo is converged {done_converge}")
                return
            self.state_action_function = state_action_func_new.copy()
            counter += 1
            if counter % self.policyparameter.decay_steps == 0:
                self.policyparameter.epsilon_greedy *= self.policyparameter.epsilon_greedy_decay
                self.logger.info(
                    f"epsilon_greedy is {self.policyparameter.epsilon_greedy}")

    def q_learning_off_policy(self, policy) -> None:
        """ implementation from algorithm q_learning_off_policy, algorithm 23 `Q-Learning` from lecture slides

        Args:
            policy (np.ndarray): behviour policy to get actions from
        """
        self.logger.info("q_learning_off_policy is starting")
        self._q_learning_off_policy_td_control(behavior_policy=policy)
        self._get_greedy_policy(state_action_function=None)

    def policy_iteration_sarsa_evalulation(self) -> None:
        """ policy iteration with sarsa evaluation for the state action function algo 22
        """
        self.logger.info(f"policy_iteration_sarsa_evalulation")
        done = False
        counter = 0
        while not done:
            old_policy = self.agent.policy.copy()
            self.logger.debug(
                f"start sarsa evaluation for state action function")
            self._sarsa_evaluate_state_action_func()
            self.logger.debug(f"use epsilon greedy for new action function")
            self._get_epsilon_greedy_improved_policy(
                state_action_function=None)
            counter += 1
            self.logger.debug(f"policy is given by {self.agent.policy}")
            if self.policyparameter.run_time_method == RunTimeMethod.CRITERION.value:
                if (np.abs(self.agent.policy - old_policy) < self.policyparameter.epsilon).all():
                    # entropies = np.apply_along_axis(
                    #     lambda x: entropy(x, base=2), axis=-1, arr=self.agent.policy.copy())
                    # self.logger.debug(f"entropies is given by {entropies}")
                    # if (entropies < self.policyparameter.epsilon).all():
                    done = True
                    self.logger.info(
                        "policy_iteration_sarsa_evalulation is converged")
                if counter % self.policyparameter.decay_steps == 0:
                    self.logger.debug(f"number of steps {counter}")
                    self.policyparameter.epsilon_greedy *= self.policyparameter.epsilon_greedy_decay
                    self.logger.info(
                        f"epsilon greedy value is {self.policyparameter.epsilon_greedy}")
            elif self.policyparameter.run_time_method == RunTimeMethod.EPISODES.value:
                if counter >= self.policyparameter.episodes:
                    done = True
                if counter % self.policyparameter.decay_steps == 0:
                    self.logger.debug(f"number of steps {counter}")
                    self.policyparameter.epsilon_greedy *= self.policyparameter.epsilon_greedy_decay
                    self.logger.info(
                        f"epsilon greedy value is {self.policyparameter.epsilon_greedy}")
            else:
                raise NotImplementedError("Method is not implemented yet")

    def _init_state_action_function(self, state_action_init: float) -> np.ndarray:
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
            raise NotImplementedError("Unknown environment type")
        return state_action_function

    def _init_state_type(self) -> None:
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
            except NotImplementedError as exc:
                self.logger.exception(str(exc))

    def _sarsa_evaluate_state_action_func(self) -> None:
        """ sarsa policy evaluation from lecture algorithm 22

        Raises:
            NotImplementedError: If observation space is not MultiDiscrete or Discrete
        """
        state_action_func_new = self._init_state_action_function(
            state_action_init=self.init_parameter.stateactionfunctioninit)

        number_of_times_played = np.zeros_like(
            self.state_action_function.copy())
        done_converge = False

        while not done_converge:

            # Sample from starting function
            self.logger.debug("Starting new trajectory")
            state = self.environment.costum_sample()
            self.logger.debug(f"Starting state is given by {state}")
            # Create trajectory given the starting state
            # action = self.agent.get_action(state)
            action = random.choice(self.environment.get_valid_actions(state))
            self.logger.debug(f"action is given by {action}")
            self.environment.reset()
            self.environment.state = state
            done = False
            while not done:
                # sample reward, next state, done
                next_state, reward, done, _ = self.environment.step(action)
                self.logger.debug(
                    f"next state is given by {next_state} and reward is {reward} and done is {done}")
                next_action = self.agent.get_action(next_state)
                self.logger.debug(f"action is given by {next_action}")
                alpha = self._get_alpha(
                    state=state, action=action, times_played=number_of_times_played)
                self.logger.debug(f"alpha is given by {alpha}")
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
                self.logger.debug(
                    f"State action function value is given by {state_action_func_new[state_pos][action]}")
                number_of_times_played[state_pos][action] += 1
                action = next_action
                state = next_state
            self.logger.debug(
                f"number  of times playes is {number_of_times_played}")
            self.logger.debug(
                f"convergence is {np.sum(np.abs(state_action_func_new - self.state_action_function))}")
            if (np.abs(state_action_func_new - self.state_action_function) < self.policyparameter.epsilon).all():
                self.logger.debug(
                    f"State action function {state_action_func_new}")
                done_converge = True
                self.logger.info(
                    f"sarsa evaluation is converged {done_converge}")
                return
            self.state_action_function = state_action_func_new.copy()

    def _get_alpha(self, state: Union[int, np.ndarray], action: int, times_played: np.ndarray, rate: float = 1.0) -> float:
        if self.state_type == 'MultiDiscrete':
            state_pos = tuple(state)
        elif self.state_type == 'Discrete':
            state_pos = state
        else:
            raise NotImplementedError(
                "Only MultiDiscrete and Discrete are supported up to now")
        return 1 / (1 + times_played[state_pos][action]) ** rate

    def _get_epsilon_greedy_improved_policy(self, state_action_function: Union[np.ndarray, None] = None) -> None:
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

    def _get_greedy_policy(self, state_action_function: Union[np.ndarray, None] = None) -> None:
        self.logger.debug(f"{self.state_action_function}")
        if state_action_function is None:
            max_indices = np.argmax(self.state_action_function, axis=-1)
        else:
            max_indices = np.argmax(state_action_function, axis=-1)

        self.agent.policy = np.zeros_like(
            self.agent.policy)
        self.agent.policy[tuple(np.indices(
            self.state_action_function.shape[:-1])) + (max_indices,)] = 1

    def _q_learning_off_policy_td_control(self, behavior_policy: np.ndarray) -> None:
        self.logger.debug(f"starting policy is given by {behavior_policy}")
        self.agent.policy = behavior_policy.copy()
        number_of_times_played = np.zeros_like(
            self.state_action_function.copy())

        self.state_action_function = self._init_state_action_function(
            state_action_init=10.0)
        state_action_func_new = self.state_action_function.copy()
        self.logger.debug(
            f"starting state action function is given by {state_action_func_new}")

        done_converge = False

        while not done_converge:
            # Sample from starting function
            state = self.environment.costum_sample()
            self.logger.debug(f"starting state is {state}")
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
                alpha = self._get_alpha(
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
                self.logger.debug(f"next state is {state}")
                self.logger.debug(50*"*")
                self.logger.debug(
                    f"state action function is given by {state_action_func_new}")
            self.logger.debug(
                f'convergence is {np.sum(np.abs(state_action_func_new - self.state_action_function))}')
            if (np.abs(state_action_func_new - self.state_action_function) < self.policyparameter.epsilon).all():
                done_converge = True
                self.logger.info(
                    f"q_learning_off_policy_td_control is converged {done_converge}")
                return
            self.state_action_function = state_action_func_new.copy()
