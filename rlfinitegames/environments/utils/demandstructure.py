"""implement probability distribution for demand in ice vendor game"""
import random
from abc import ABC, abstractmethod
from math import exp, factorial

#TODO: implement further probability distribution for demand in ice vendor game
class BaseDemand(ABC):
    """Base class for probability distribution for demand in ice vendor game"""
    def __init__(self, max_inventory) -> None:
        self.max_inventory = max_inventory

    @abstractmethod
    def pmf(self, k_int: int) -> float:
        """get probability distribution for demand in ice vendor game"""

    @abstractmethod
    def cdf(self, k_int: int) -> float:
        """get cumulative distribution function for demand in ice vendor game"""

    @abstractmethod
    def sample(self) -> int:
        """sample from probability distribution for demand in ice vendor game"""

class PoissonRandomVariable(BaseDemand):
    """ poisson random variable with support for max_inventory """
    def __init__(self, max_inventory: int, lam: float) -> None:
        super().__init__(max_inventory)
        self.lam = lam
        self.scaling_factor = sum((exp(-self.lam) * (self.lam ** k_int) / factorial(
            k_int) for k_int in range(0, self.max_inventory + 1)))
        self.prob_vector = [self.pmf(k_int)
                            for k_int in range(0, self.max_inventory + 1)]

    def pmf(self, k_int: int) -> float:
        """get probability distribution function for demand in ice vendor game"""
        if k_int > self.max_inventory:
            return 0.0
        return exp(-self.lam) * (self.lam ** k_int) / factorial(k_int) / self.scaling_factor

    def cdf(self, k_int: int) -> float:
        """ calculate cumulative distribution function
        """
        if k_int >= self.max_inventory:
            return 1.0
        cdf = 0.0
        for i in range(k_int + 1):
            cdf += self.pmf(i)
        return cdf

    def sample(self) -> int:
        """ sample from Poisson distribution
        """
        rand_float = random.random()
        cum = 0
        for index, probability in enumerate(self.prob_vector):
            cum += probability
            if rand_float < cum:
                break
        return index


class BinomialRandomVariable(BaseDemand):
    """ binomial random variable with support for max_inventory """
    def __init__(self, max_inventory: int, _p: float) -> None:
        super().__init__(max_inventory=max_inventory)

    def pmf(self, _k_int: int) -> float:
        pass

    def cdf(self, _k_int: int) -> float:
        pass


class NegativeBinomialRandomVariable():
    """ negative binomial random variable with support for max_inventory """
    def __init__(self, max_inventory: int, _p: float) -> None:
        super().__init__(max_inventory=max_inventory)

    def pmf(self, k_int: int) -> float:
        """get probability distribution function for demand in ice vendor game"""

    def cdf(self, k_int: int) -> float:
        """get cumulative distribution function for demand in ice vendor game"""
