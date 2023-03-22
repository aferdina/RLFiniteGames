import random 
from math import exp, factorial


class PoissonRandomVariable():
    def __init__(self, max_inventory: int, lam: float) -> None:
        self.max_inventory = max_inventory
        self.lam = lam
        self.scaling_factor = sum([exp(-self.lam) * (self.lam ** k_int) / factorial(
            k_int) for k_int in range(0, self.max_inventory + 1)])
        self.prob_vector = [self.pmf(k_int)
                            for k_int in range(0, self.max_inventory + 1)]

    def pmf(self, k_int: int) -> float:
        if k_int > self.max_inventory:
            return 0.0
        else:
            return exp(-self.lam) * (self.lam ** k_int) / factorial(k_int) / self.scaling_factor

    def cdf(self, k_int: int) -> float:
        """ calculate cumulative distribution function
        """
        if k_int >= self.max_inventory:
            return 1.0
        else:
            cdf = 0.0
            for i in range(k_int + 1):
                cdf += self.pmf(i)
            return cdf

    def sample(self) -> int:
        """ sample from Poisson distribution
        """
        rand_float = random.random()
        cum = 0
        for i, p in enumerate(self.prob_vector):
            cum += p
            if rand_float < cum:
                break
        return i


class BinomialRandomVariable():
    def __init__(self, max_inventory: int, p: float) -> None:
        pass

    def pmf(self, k_int: int) -> float:
        pass

    def cdf(self, k_int: int) -> float:
        pass


class NegativeBinomialRandomVariable():
    def __init__(self, max_inventory: int, p: float) -> None:
        pass

    def pmf(self, k_int: int) -> float:
        pass

    def cdf(self, k_int: int) -> float:
        pass
