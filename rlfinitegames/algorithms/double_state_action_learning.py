""" Implementation of various algorithms based on Robbins Siegmund. 
"""
from dataclasses import dataclass, field
import random
import logging
from itertools import product
from typing import Union
import numpy as np
from gym import Env, spaces
from rlfinitegames.logging_module.setup_logger import setup_logger
from rlfinitegames.policies.discrete_agents import FiniteAgent
from rlfinitegames.algorithms.helperclasses import (RunTimeParameter,
                                                    PolicyMethod,
                                                    UpdateMethod,
                                                    RunTimeMethod,
                                                    PolicyMethodNames,
                                                    RobbingsSiegmundUpdate)

# statics for logging purposes
LOGGINGPATH = "rlfinitegames/logging_module/logfiles/"
FILENAME = "double_state_action_learning"
LOGGING_LEVEL = logging.DEBUG
LOGGING_CONSOLE_LEVEL = logging.INFO
LOGGING_FILE_LEVEL = logging.DEBUG

INVALIDSTATEINIT = -1000000.0

# TODO: Create a running files for the different games
# TODO: can we initialize docstrings by using the field method from dataclasses?


@dataclass
class InitConfig:
    """ Dataclass to specify the initialization values for the stateaction function in the algorithm
    """
    stateactionfunctioninit: float = field(default=0.0, metadata={
        "doc": "initialisation value for the stateaction function"},
        init=True)
    invalidstateactionvalue: float = field(default=INVALIDSTATEINIT, metadata={
                                           "doc":
                                           "initialisation value for invalid state action values"},
                                           init=True)
    number_of_action_functions: int = field(default=2, metadata={
                                            "doc":
                                            "number of state action functions to initialize"},
                                            init=True)


@dataclass
class DoubleParameter:
    """ class to represent all configurations of robbins siegmund algorithms
    """
    gamma: float = field(default=0.95, metadata={
                         "doc": "discount factor"})
    initialisation: InitConfig = field(
        default_factory=lambda: InitConfig(stateactionfunctioninit=0.0,
                                           invalidstateactionvalue=INVALIDSTATEINIT,
                                           number_of_action_functions=2),
        metadata={"doc":
                  "initialization parameter for state action function"})
    epsilon_greedy: float = field(
        default=0.1, metadata={"doc": "epsilon greedy parameter"})
    rate: float = field(default=1.0, metadata={
                        "doc": "rate parameter for step size"})
    runtimemethod: RunTimeParameter = field(default_factory=lambda: RunTimeParameter(
        run_time_method=RunTimeMethod.EPISODES.value, episodes=1000))
    updatemethod: RobbingsSiegmundUpdate = field(
        default_factory=lambda: RobbingsSiegmundUpdate(update_method=UpdateMethod.PLAIN.value))
    policy_method: PolicyMethod = field(default_factory=lambda: PolicyMethod())


class DoubleStateActionLearning():
    """ main class for implementing double state action learning algorithm
    """

    def __init__(self, environment: Union[Env, str],
                 policy: FiniteAgent,
                 algo_params: DoubleParameter):
        self.logger = setup_logger(logger_name=__name__,
                                   logger_level=LOGGING_LEVEL,
                                   log_file=LOGGINGPATH + FILENAME + ".log",
                                   file_handler_level=LOGGING_FILE_LEVEL,
                                   stream_handler_level=LOGGING_CONSOLE_LEVEL,
                                   console_output=True)
        self.policyparameter = algo_params  # policy evaluation parameter
        self.environment = environment  # environment class
        self.agent = policy  # agent class

        # initialize state type
        self.state_type = None
        self._init_state_type()

        self.state_action_functions = [self._init_state_action_function(
            self.policyparameter.initialisation.stateactionfunctioninit) for _ in range(
            self.policyparameter.initialisation.number_of_action_functions)]

    def _init_state_type(self) -> None:
        """ initializes the state type of the gym environment

        Raises:
            NotImplementedError: Up to now only
            `Discrete` and `Multidiscrete` gym observation spaces are supported
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

    def _init_state_action_function(self, state_action_init: float) -> np.ndarray:
        state_action_function = np.ones_like(
            self.agent.policy) * state_action_init
        # TODO: what about terminal states?
        if isinstance(self.environment.observation_space, spaces.MultiDiscrete):
            # TODO: not nice implementation yet
            for state in list(
                    product(
                        *[list(
                            range(
                                element))
                          for element in
                          self.environment.observation_space.nvec.tolist(
                        )])):
                state_pos_action = self.environment.get_valid_actions(
                    state)
                not_pos_actions = set(self.agent.all_actions) - \
                    set(state_pos_action)
                state_action_function[state][list(
                    not_pos_actions)] = self.policyparameter.initialisation.invalidstateactionvalue
        elif isinstance(self.environment.observation_space, spaces.Discrete):
            for state in range(self.environment.observation_space.n):
                state_pos_action = self.environment.get_valid_actions(
                    state)
                not_pos_actions = set(self.agent.all_actions) - \
                    set(state_pos_action)
                state_action_function[state][list(
                    not_pos_actions)] = self.policyparameter.initialisation.invalidstateactionvalue
        else:
            raise NotImplementedError("Unknown environment type")
        return state_action_function

    def _get_greedy_policy(self, state_action_function: Union[np.ndarray, None] = None) -> None:
        self.logger.debug(f"{self.state_action_functions[0]}")
        if state_action_function is None:
            max_indices = np.argmax(self.state_action_functions[0], axis=-1)
        else:
            max_indices = np.argmax(state_action_function, axis=-1)

        self.agent.policy = np.zeros_like(
            self.agent.policy)
        self.agent.policy[tuple(np.indices(
            self.state_action_functions[0].shape[:-1])) + (max_indices,)] = 1

    def _get_alpha(self,
                   state: Union[int, np.ndarray],
                   action: int, times_played: np.ndarray,
                   rate: float = 1.0) -> float:
        if self.state_type == 'MultiDiscrete':
            state_pos = tuple(state)
        elif self.state_type == 'Discrete':
            state_pos = state
        else:
            raise NotImplementedError(
                "Only MultiDiscrete and Discrete are supported up to now")
        return 1 / (1 + times_played[state_pos][action]) ** rate

    def run_double_state_action_learning(self) -> None:
        """ running different double state action learning methods from lecture
        """
        self._double_state_action_learning()
        self._get_greedy_policy()

    def _double_state_action_learning(self) -> None:
        counter = 0
        self.logger.info("double state action function learning")
        number_of_times_played = np.zeros_like(
            self.state_action_functions[0].copy())
        done_converge = False

        state_action_functions = [policy.copy()
                                  for policy in self.state_action_functions]
        while not done_converge:
            self.logger.debug("Creating a new trajectory")
            # Sample from starting function
            state = self.environment.costum_sample()
            # get action depending on policy method
            self.environment.reset()
            self.environment.state = state
            done = False
            while not done:
                if self.policyparameter.policy_method.method_name == PolicyMethodNames.BEHAVIOUR.value:
                    action = self.agent.get_action(state)
                elif self.policyparameter.policy_method.method_name == PolicyMethodNames.EPSILONGREEDY.value:
                    summed_state_action_function = np.add.reduce(
                        self.state_action_functions)
                    self._get_greedy_policy(
                        state_action_function=summed_state_action_function)
                    action = self.agent.get_action(state)
                else:
                    raise ValueError("Unknown policy method")
                # sample reward, next state, done
                next_state, reward, done, _ = self.environment.step(action)
                self.logger.debug(
                    f"next_state is {next_state}, reward is {reward} and done is {done}")
                alpha = self._get_alpha(
                    state=state,
                    action=action,
                    times_played=number_of_times_played,
                    rate=self.policyparameter.rate)
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

                # generate random number to determine which state action function should be updated
                index_set = set(
                    list(range(self.policyparameter.initialisation.number_of_action_functions)))
                updated_function_pos = random.choice(list(index_set))
                # drop used index to get state action function for estimating the maximum value
                index_set.remove(updated_function_pos)
                # TODO: add updating all state action functions at once
                # update state aciton function given different methods
                if self.policyparameter.updatemethod.update_method == UpdateMethod.PLAIN.value:
                    update_value = reward + self.policyparameter.gamma *\
                        state_action_functions[random.choice(list(
                            index_set))][next_state_pos][np.argmax(
                                state_action_functions[updated_function_pos][next_state_pos])]
                elif self.policyparameter.updatemethod.update_method == UpdateMethod.TRUNCUATED.value:
                    update_value = reward + self.policyparameter.gamma * \
                        state_action_functions[updated_function_pos][next_state_pos][np.argmax(
                            state_action_functions[updated_function_pos][next_state_pos])] + \
                        self.policyparameter.gamma * \
                        self.policyparameter.updatemethod.trunc_bounds.trunc_value(np.min(
                            state_action_functions[random.choice(list(
                                index_set))][next_state_pos][np.argmax(
                                    state_action_functions[updated_function_pos][next_state_pos])] -
                            state_action_functions[random.choice(list(
                                index_set))][next_state_pos][np.argmax(
                                    state_action_functions[updated_function_pos][next_state_pos])]), factor=alpha)
                elif self.policyparameter.updatemethod == UpdateMethod.CLIPPED.value:
                    update_value = reward + self.policyparameter.gamma * \
                        np.minimum(
                            state_action_functions[updated_function_pos][next_state_pos][np.argmax(
                                state_action_functions[updated_function_pos][next_state_pos])],
                            state_action_functions[random.choice(list(
                                index_set))][next_state_pos][np.argmax(
                                    state_action_functions[updated_function_pos][next_state_pos])])
                else:
                    raise ValueError(
                        "Invalid method for updating the state action function")
                state_action_functions[updated_function_pos][state_pos][action] = (
                    state_action_functions[updated_function_pos][state_pos][action] +
                    alpha * (
                        update_value -
                        state_action_functions[updated_function_pos][state_pos][action]))
                number_of_times_played[state_pos][action] += 1
                state = next_state

                # some loggings for debugging
                self.logger.debug(f"next state is {state}")
                self.logger.debug(50*"*")

            counter += 1
            self.logger.debug(f"counter is {counter}")
            if self.policyparameter.runtimemethod == RunTimeMethod.CRITERION.value:
                if sum(np.sum(np.abs(
                        state_action_new - state_action_old)) for (
                            state_action_new, state_action_old) in zip(
                        state_action_functions,
                        self.state_action_functions)) < self.policyparameter.runtimemethod.epsilon:
                    done_converge = True
                    self.logger.debug("convergence is reached")
            elif self.policyparameter.runtimemethod == RunTimeMethod.EPISODES.value:
                if counter >= self.policyparameter.runtimemethod.episodes:
                    done_converge = True
            self.state_action_functions = [
                arr.copy() for arr in state_action_functions]
# TODO: init terminal positions for q values


if __name__ == "__main__":
    # print(DoubleParameter.__doc__)
    # print(FiniteAgent.__doc__)
    pass
