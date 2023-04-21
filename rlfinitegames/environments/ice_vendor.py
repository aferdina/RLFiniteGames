""" Implementation of Ice Vendor example from lecture"""
from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum
from gym import spaces, Env
# pylint: disable=C0301
from rlfinitegames.environments.utils.demandstructure import PoissonRandomVariable, BinomialRandomVariable, NegativeBinomialRandomVariable
from rlfinitegames.environments.utils.helpfunctions import rgetattr

# we start with an empty storage
START_STATE = 0
MAX_STEPS = 100


class DemandStructure(Enum):
    """ Different type of Demand structures """
    POISSON = PoissonRandomVariable
    BINOMIAL = BinomialRandomVariable
    NEGATIVE_BINOMIAL = NegativeBinomialRandomVariable


@dataclass
class GameConfig:
    """ Game configuration
    """
    max_inventory: int = 20  # maximum inventory
    production_cost: float = 2.0  # production cost for ice cream production
    storage_cost: float = 1.0  # storage cost for ice cream over night
    selling_price: float = 5.0  # selling price for ice cream over night
    demand_structure: DemandStructure = "POISSON"  # demand structure
    demand_parameters: Dict[str, int] = None  # parameter of demand structure


class IceVendor(Env):
    """ Ice Vendor Environment
    """

    def __init__(self, game_config: GameConfig) -> None:
        """ initialize ice vendor environment

        Args:
            game_config (GameConfig): including game configuration like selling price, demand structure
        """

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
        self.max_steps = None
        self.reset()

    def step(self, action: int) -> list[int, float, bool, dict]:
        """ run a step in the ice vendor game

        Args:
            action (int): amount of ice cream to buy

        Returns:
            list[int, float, bool, dict]: next state, reward, information if finish, and info
        """

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
        """ calculate the storage costs for a given state

        Args:
            state (int): amount of ice cream in the storage

        Returns:
            float: storage costs for the given state
        """
        return self.game_config.storage_cost * state

    def calculate_production_cost(self, action: int) -> float:
        """ calculate production cost given action played"""
        return self.game_config.production_cost * action

    def calculate_selling_price(self, sold_products: int) -> float:
        """ calculates the selling price given the sold products

        Args:
            sold_products (int): amount of sold products

        Returns:
            float: cash making by sold products
        """
        return self.game_config.selling_price * sold_products

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> None:
        """ reset the game

        Args:
            seed (Optional[int], optional): Random seed for the game. Defaults to None.
            options (Optional[dict], optional): Some options i dont even know. Defaults to None.
        """
        # TODO: Can someone explain Andre the options attribute from the base class
        super().reset(seed=seed)
        self.max_steps = MAX_STEPS
        self.timestep = 0
        self.state = START_STATE

    def calculate_probability(self, state: int, action: int) -> List[float]:
        """calculate the probability to get in the next by taking action in specific state

        Args:
            state (int): amount of ice cream in the storage
            action (int): amount of ice cream bought

        Returns:
            List[float]: Probability vector for all of the next states
        """
        # create a list for all next states
        prob_next_state = [0.0 for _ in range(
            self.game_config.max_inventory + 1)]

        # calculat the next state without considering any demand
        next_state_without_demand = min(
            state + action, self.game_config.max_inventory)
        # calculte the probability of a next state given the formular of the lecture
        prob_next_state[0] = (sum(self.demand_structure.pmf(
            next_state_without_demand + i) for i in range(self.game_config.max_inventory - next_state_without_demand + 1)))
        for index in range(1, next_state_without_demand + 1):
            prob_next_state[index] = self.demand_structure.pmf(
                next_state_without_demand - index)
        return prob_next_state

    def get_rewards(self, state: int, action: int) -> List[float]:
        """get the rewards for a given state and action

        Args:
            state (int): amount of ice cream in the storage
            action (int): amount of ice cream bought

        Returns:
            list[float]: list of rewards for all possible next states
        """
        rewards = []
        for next_state in range(self.game_config.max_inventory + 1):
            # what is the number of sold products
            # calculating reward
            reward = self.calculate_selling_price(
                state + action - next_state) - self.calculate_storage_cost(next_state) - self.calculate_production_cost(action)
            rewards.append(reward)
        return rewards

    def render(self, _mode: str = "human") -> None:
        """ rendering the game

        Args:
            _mode (str, optional): mode to render the game. Defaults to "human".
        """
        print(self.info)

    def get_valid_actions(self, state: int) -> list[int]:
        """ get the valid actions for a given state

        Args:
            state (int): amount of ice cream in the storage

        Returns:
            list[int]: list with all valid actions for the given state
        """
        valid_actions = list(range(
            self.game_config.max_inventory + 1 - state))
        return valid_actions

    def costum_sample(self) -> int:
        """ get a random sample from the game state equiv storage

        Returns:
            int: potential storage state
        """
        return self.observation_space.sample()
