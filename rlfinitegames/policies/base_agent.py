""" initialize base class for Finite Agents"""
from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np

# pylint: disable=too-few-public-methods


class BaseAgent(ABC):
    """base class for all agents"""

    def __init__(self, seed: Optional[int] = None, masking: bool = True):
        self.rng = np.random.default_rng(seed=seed)
        self.masking = masking

    @abstractmethod
    def get_action(self, state: Union[np.array, int]) -> int:
        """get action for given state"""

    @abstractmethod
    def _update_action_mask_prob(self, state: Union[np.array, int]) -> int:
        """update action given state"""
