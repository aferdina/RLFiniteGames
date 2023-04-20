# Reinforcement Learning Learning Programming Task 06

## Task 1: Implement the Ice Vendor example from the lecture

### How to call the function?

The Ice Vendor example is implemented in [here](../environments/ice_vendor.py).

The game can be called and played as follows:

```sh
python rlfinitegames/play_games/run_ice_vendor.py
```

### Some Comments to the game implementation

The ice Vendor example is starting in the state `START_STATE`. In my implementation, the Iceman starts without Ice Cream in his stock.

```python
# we start with an empty storage
START_STATE = 0
```

We want to keep the implementation as generic as possible and therefore allow different distributions for the demand. Feel free to support me and implement more distributions.

```python
class DemandStructure(Enum):
    """ Different type of Demand structures """
    POISSON = PoissonRandomVariable
    BINOMIAL = BinomialRandomVariable
    NEGATIVE_BINOMIAL = NegativeBinomialRandomVariable
```
