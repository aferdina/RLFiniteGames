""" Implementation of Ice Vendor example from lecture"""
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
from gym import spaces, Env
# pylint: disable=C0301
from rlfinitegames.environments.utils.demandstructure import PoissonRandomVariable, BinomialRandomVariable, NegativeBinomialRandomVariable
from rlfinitegames.environments.utils.helpfunctions import rgetattr

START_STATE = 0

class DemandStructure(Enum):
    """ Different type of Demand structures """
    POISSON = PoissonRandomVariable
    BINOMIAL = BinomialRandomVariable
    NEGATIVE_BINOMIAL = NegativeBinomialRandomVariable


@dataclass
class GameConfig:
    """ Game configuration
    """
    max_inventory: int = 20
    production_cost: float = 2.0
    storage_cost: float = 1.0
    selling_price: float = 5.0
    demand_structure: DemandStructure = "POISSON"
    demand_parameters: Dict[str, int] = None
    mean_no_days: int = 20


class IceVendor(Env):
    """ Ice Vendor Environment """

    def __init__(self, game_config: GameConfig):

        # initialize the game config
        self.game_config = game_config

        # initialize action and observation spaces
        self.action_space = spaces.Discrete(
            self.game_config.max_inventory + 1)
        self.observation_space = spaces.Discrete(
            self.game_config.max_inventory + 1)

        # initialize the demand structure
        self.demand_structure = rgetattr(DemandStructure, f"{self.game_config.demand_structure}.value")(
            max_inventory=self.game_config.max_inventory, **self.game_config.demand_parameters)

        # initialize information of game
        self.info = {}
        # reset the game
        self.timestep = None
        self.state = None
        self.reset()


    def step(self, action: int) -> list[int, float, bool, dict]:
        """ run one step in the environment """

        # getting demand
        demand = self.demand_structure.sample()

        # getting next state
        _next_state = max(self.state + action - demand, 0)
        next_state = min(_next_state, self.game_config.max_inventory)

        # calculating reward
        reward = self.calculate_selling_price(
            self.state + action - next_state) - self.calculate_storage_cost(next_state) - self.calculate_production_cost(action)

        info = {"demand": demand, "next_state": next_state, "sold_items": self.state + action - next_state, "money_made": self.calculate_selling_price(
            self.state + action - next_state), "storage_cost": self.calculate_storage_cost(next_state), "production_cost": self.calculate_production_cost(action)}
        self.info = info
        self.state = next_state

        done = False
        self.timestep += 1
        if self.timestep >= self.max_steps:
            done = True
        return next_state, reward, done, info

    def calculate_storage_cost(self, state: int) -> float:
        """ calculate storage cost given state"""
        return self.game_config.storage_cost * state

    def calculate_production_cost(self, action: int) -> float:
        """ calculate production cost given action played"""
        return self.game_config.production_cost * action

    def calculate_selling_price(self, sold_products: int) -> float:
        """calculates the selling price given the sold products"""
        return self.game_config.selling_price * sold_products

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> None:
        super().reset(seed=seed)
        self.max_steps = 100
        self.timestep = 0
        self.state = START_STATE

    def calculate_probability(self, state: int, action: int) -> float:
        """calculate the probability to get in the next by taking action in specific state"""
        prob_next_state = [0 for _ in range(
            self.game_config.max_inventory + 1)]

        next_state_without_demand = min(
            state + action, self.game_config.max_inventory)
        prob_next_state[0] = (sum(self.demand_structure.pmf(
            next_state_without_demand + i) for i in range(self.game_config.max_inventory - next_state_without_demand + 1)))
        for index in range(1, next_state_without_demand + 1):
            prob_next_state[index] = self.demand_structure.pmf(
                next_state_without_demand - index)
        return prob_next_state

    def get_rewards(self, state: int, action: int) -> list[float]:
        """get the rewards for a given state and action"""
        rewards = []
        for next_state in range(self.game_config.max_inventory + 1):
            #what is the number of sold products
            # calculating reward
            reward = self.calculate_selling_price(
                state + action - next_state) - self.calculate_storage_cost(next_state) - self.calculate_production_cost(action)
            rewards.append(reward)
        return rewards


    def render(self, _mode: str = "human"):
        print(self.info)

    def get_valid_actions(self, state: int) -> list[int]:
        """get the valid actions for a given state"""
        valid_actions = list(range(
            self.game_config.max_inventory + 1 - state))
        return valid_actions

    def costum_sample(self) -> int:
        """ get a random sample from the game state equiv storage

        Returns:
            int: potential storage state
        """
        return self.observation_space.sample()
