""" base class for algorithms
"""
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np


class PolicyMethodNames(Enum):
    """ enumeration of policy methods for double q learning
    """
    EPSILONGREEDY = "epsilongreedy"
    BEHAVIOUR = "behaviour"


class UpdateMethod(Enum):
    """ An enumeration of update methods"""
    PLAIN = "plain"
    TRUNCUATED = "truncuated"
    CLIPPED = "clipped"


class PolicyIterationApproaches(Enum):
    """ Enumeration of all possible policy iteration approaches"""
    NAIVE = 'Naive'
    SWEEP = 'Sweep'


class MonteCarloApproaches(Enum):
    """ Enumeration of all possible policy iteration approaches"""
    STATE_ACTION_FUNCTION = 'StateActionFunction'
    VALUE_FUNCTION = 'ValueFunction'


@dataclass
class MonteCarloPolicyIterationParameters:
    """ Parameter for MonteCarloPolicyIteration
    """
    montecarloapproach: MonteCarloApproaches
    valuefunctioninit: float | None = None
    stateactionfunctioninit: float | None = None
    invalidstateactionvalue: float | None = None


@dataclass
class PolicyIterationParameter:
    """
    Class for Policy Evaluation for Naive or Sweep Approach
    """
    epsilon: float = 0.01
    gamma: float = 0.95
    approach: PolicyIterationApproaches = PolicyIterationApproaches.NAIVE
    epsilon_greedy: float = 0.1
    epsilon_greedy_decay: float = 1.0
    decay_steps: int = 1


class RunTimeMethod(Enum):
    """ Dataclass to specify the method to run the algorithm. If episodes is used, then 
    the algorithm is running until a number of episodes have been completed. Otherwise,
    the algorithm
    is running until a convergence criterion has been reached.
    """
    EPISODES = "episodes"
    CRITERION = "criterion"


@dataclass
class RunTimeParameter:
    """ parameter for runtime method in double q learning
    """
    run_time_method: RunTimeMethod = field(
        default=RunTimeMethod.EPISODES.value)
    episodes: int | None = None
    epsilon: float | None = None

    def __post_init__(self):
        if self.run_time_method == RunTimeMethod.EPISODES.value:
            if self.episodes is None:
                raise ValueError(
                    "Episodes must be specified if runtime method is Episodes")
        if self.run_time_method == RunTimeMethod.CRITERION.value:
            if self.epsilon is None:
                raise ValueError(
                    "Epsilon must be specified if runtime method is criterion")


@dataclass
class TruncatedBounds(ABC):
    """ class for truncated method in double q learning"""
    lower_bound: float = field(default=10.0, metadata={
                               "description": "lower bound for truncation"})
    upper_bound: float = field(default=10.0, metadata={
                               "description": "upper bound for truncation"})

    @abstractmethod
    def trunc_value(self, a_value: float, factor: float) -> float:
        """ return truncated value given bounds

        Args:
            a_value (float): value to truncate
            factor (float): factor for truncation bounds

        Returns:
            float: clipped value
        """
        return np.clip(a=a_value, a_max=self.upper_bound * factor, a_min=-self.lower_bound * factor)


@dataclass
class RobbingsSiegmundUpdate:
    """parameter class for robbings siegmund update"""
    update_method: UpdateMethod = field(default=UpdateMethod.PLAIN.value)
    trunc_bounds: TruncatedBounds | None = None

    def __post_init__(self):
        if self.update_method == UpdateMethod.TRUNCUATED.value:
            if self.trunc_bounds is None:
                raise ValueError(
                    "bounds must be implemented if method is TRUNCUATED")


@dataclass
class PolicyMethod:
    method_name: PolicyMethodNames = PolicyMethodNames.BEHAVIOUR.value
    epsilon_greedy: float | None = None

    def __post_init__(self):
        if self.method_name == PolicyMethodNames.EPSILONGREEDY:
            if self.epsilon_greedy is None:
                raise ValueError(
                    "epsilon_greedy must be implemented if method is EPSILONGREEDY")
