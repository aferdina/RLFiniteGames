from enum import Enum
from typing import Union
from dataclasses import dataclass

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
    valuefunctioninit: Union[float, None] = None
    stateactionfunctioninit: Union[float, None] = None
    invalidstateactionvalue: Union[float, None] = None


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

class PolicyMethod(Enum):
    EPSILONGREEDY = "epsilongreedy"
    BEHAVIOUR = "behaviour"