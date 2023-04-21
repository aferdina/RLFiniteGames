# Reinforcement Learning Learning Programming Task 06

## Task 1: Implement the Ice Vendor example from the lecture

### How to call the function?

The Ice Vendor example is implemented in [here](../../environments/ice_vendor.py).

The game can be called and played as follows:

```sh
python rlfinitegames/play_games/run_ice_vendor.py
```

***

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

***

**Andre**: What are the advantages of the `enum` class in python?

**ChatGPT**:
Here are some advantages of using `enum` classes in Python:

1. Improved code clarity: `enum` classes make code more self-documenting by providing meaningful names to represent a value or a group of values.

2. Type safety: `enum` classes allow type-checking at compile-time or run-time to ensure that only valid enumeration values are used in the code.

3. Namespace control: `enum` classes allow you to define your own namespaces for enumeration values to avoid naming collisions with other variables or constants in your code.

4. Introspection: `enum` classes provide a built-in way to inspect the values and members of an enumeration at run-time.

5. Serialization: `enum` classes can be serialized and deserialized to and from strings or integers, making them useful for storing configuration data or communicating with external systems.

6. Enum members can have metadata attached: Enum members can have attached attributes, that can be used to provide additional information to the member, e.g. documentation, labels, or values.

Overall, `enum` classes provide a cleaner, more organized, and less error-prone way to define symbolic names in your Python code.

***

Next, a configuration class is set up for the game. This contains all parameters that can be adjusted.

```python
@dataclass
class GameConfig:
    """ Game configuration
    """
    max_inventory: int = 20 # maximum inventory
    production_cost: float = 2.0 # production cost for ice cream production
    storage_cost: float = 1.0 # storage cost for ice cream over night
    selling_price: float = 5.0 # selling price for ice cream over night
    demand_structure: DemandStructure = "POISSON" # demand structure
    demand_parameters: Dict[str, int] = None # parameter of demand structure
```

***

Next, we consider the components of the Markov decision problem.

The action space is given by the amount of ice that can be purchased. The inventory is limited by the parameter `self.game_config.max_inventory`. Thus the action space is given as follows:

```python
self.action_space = spaces.Discrete(
            self.game_config.max_inventory + 1)
```

***

**Andre**: What is the spaces module from the gym environment in python?

**ChatGPT**: The `spaces` module is a part of the OpenAI Gym library for creating and working with reinforcement learning environments in Python. It provides a set of classes for defining the observation space and action space of an environment, which are used by the Gym API to specify the format of inputs and outputs for the agent.

The `spaces` module contains several classes for defining different types of observation and action spaces, including:

1. `Box`: Represents a continuous space with bounds for each dimension.

2. `Discrete`: Represents a discrete space with a fixed number of values.

3. `Tuple`: Represents a combination of other spaces, usually used to represent a composite observation or action.

4. `MultiBinary`: Represents a binary space of fixed size.

5. `MultiDiscrete`: Represents a discrete space of multiple subspaces with different sizes.

These classes define the format of the inputs and outputs for the environment and are used by the Gym API to ensure that the agent's inputs and outputs are of the correct format and shape.

Here's an example of how to create an observation space using the `Box` class:

```python
import gym
from gym import spaces

class MyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
```

In this example, the `MyEnv` environment has an observation space defined using the `Box` class. The `Box` constructor takes the following arguments:

- `low`: A scalar or array specifying the lower bounds of the space.

- `high`: A scalar or array specifying the upper bounds of the space.

- `shape`: A tuple specifying the shape of the space.

- `dtype`: The data type of the space.

In this case, the `Box` object defines a 3x3 image with pixel values between 0 and 255.

The `action_space` is defined using the `Discrete` class, which creates a space of four discrete actions.

Overall, the `spaces` module in the Gym library provides a flexible and intuitive way to define observation and action spaces for your reinforcement learning environments.

**Andre**: Which are the most important methods from the spaces class?
**ChatGPT**: The `spaces` module in the OpenAI Gym library for creating and working with reinforcement learning environments in Python provides a set of classes for defining the observation space and action space of an environment. Here are some of the most important methods of the classes in the `spaces` module:

1. `sample()`: This method generates a random sample from the space. The `Box` and `Discrete` classes have this method implemented.

2. `contains(x)`: This method checks if a given value is a valid element of the space. The `Box`, `Discrete`, `MultiBinary`, and `MultiDiscrete` classes have this method implemented.

3. `shape`: This attribute returns the shape of the space. The `Box` and `Tuple` classes have this attribute.

4. `n`: This attribute returns the number of discrete actions in the space. The `Discrete` class has this attribute.

5. `low` and `high`: These attributes return the lower and upper bounds of the space, respectively. The `Box` class has these attributes.

6. `dtype`: This attribute returns the data type of the space. The `Box` class has this attribute.

Here's an example of how to use some of these methods:

```python
import gym
from gym import spaces

# Define a Box space for a 2D point with coordinates (x, y)
point_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)

# Generate a random point from the space
random_point = point_space.sample()

# Check if the point is valid
if point_space.contains(random_point):
    print(f"Generated point: {random_point}")
else:
    print("Invalid point")

# Print the shape of the space
print(f"Shape of point space: {point_space.shape}")

# Define a Discrete space for an action with 3 possible values
action_space = spaces.Discrete(3)

# Print the number of discrete actions in the space
print(f"Number of actions: {action_space.n}")
```

In this example, we create a `Box` space for a 2D point with coordinates (x, y) that are bounded between -1.0 and 1.0. We generate a random point from the space using the `sample()` method, and then check if the point is a valid element of the space using the `contains()` method. We print the shape of the space using the `shape` attribute.

We also define a `Discrete` space for an action with 3 possible values using the `Discrete` class, and print the number of discrete actions in the space using the `n` attribute.

***

The observation space in the Ice Vendor example is meant to represent the amount of ice that is in stock. Therefore, the observation space is given as follows:

```python
self.observation_space = spaces.Discrete(
            self.game_config.max_inventory + 1)
```

In certain states not all actions are allowed. Therefore, similar to the example of Grid World, we need a mapping that gives us the set of valid actions for a state.

```python
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
```

***

Next, we deal with the rewards in the game. The ice cream man is supposed to make his decisions based on inventory costs, manufacturing costs, and selling prices. Therefore, we define mappings that perform these calculations for us.

```python
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
```

***

For the model based methods, the transition probabilities and reward functions are still needed for the game. These are calculated as follows:

```python
def calculate_probability(self, state: int, action: int) -> List[float]:
    """calculate the probability to get in the next by taking action in specific state

    Args:
        state (int): amount of ice cream in the storage
        action (int): amount of ice cream bought

    Returns:
        List[float]: Probability vector for all of the next states
    """
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
```

***

Finally, let's look at what a step looks like in the game.
Basically, a step consists of 3 parts. First, we buy a certain amount of ice cream (`action`). After the ice cream is purchased, we create a realization of the random variable that describes the demand. After the demand and the purchased amount of ice cream are fixed, we can determine the inventory for the next period and calculate all costs and revenues.

```python
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
```
