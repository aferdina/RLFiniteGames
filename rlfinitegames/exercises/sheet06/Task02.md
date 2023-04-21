# Reinforcement Learning Learning Programming Task 06

## Task 2: Use the policy iteration from the last lecture to get the optimal decision rule for the iceman

### How to call the solution?

I will eventually create a nicer module where different games can be solved using policy iteration. Unfortunately, I absolutely don't have the time right now. Maybe someone of the dedicated readers can help me with that.

TODO: Write a generic modul to run all MDP which supporting the Policy Iteration Algorithm.

```sh
python rlfinitegames/play_games/run_ice_vendor.py
```

***

### Some commments on the implementation of policy iteration

In the following, I will discuss the implementation of the Policy Iteration [module](../../algorithms/policy_iteration.py).

First, a set of parameters is set for policy iteration.

```python
# define all possible policy iteration approaches
# TODO: add Sweep Approach from lecture
class PolicyIterationApproaches(Enum):
    """ Enumeration of all possible policy iteration approaches"""
    NAIVE = 'Naive'
    SWEEP = 'Sweep'
```

In the lecture we talked about 2 approaches. On the one hand the naive approach and on the other hand the sweep approach. So far only the naive approach has been implemented, so supporters are welcome to implement the sweep approach again.

Additional parameters for policy iteration are set in the following class.

```python
@dataclass
class PolicyIterationParameter:
    """
    Class for Policy Evaluation for Naive or Sweep Approach

    :param approach: String that specifies which approach to use (Naive or Sweep)
    :param epsilon: float variable that determines termination criterium
    :param gamma: float that represents the discount factor
    """
    epsilon: float
    gamma: float
    approach: PolicyIterationApproaches
```

***

In the first step we initialize the class.

```python
def __init__(self, environment: Union[Env, str] = GridWorld(5),
                policy=FiniteAgent(),
                policyparameter: PolicyIterationParameter = PolicyIterationParameter(
                approach='Naive',
                epsilon=0.01,
                gamma=0.9),
                verbose: int = 0) -> None:
    # TODO: adding sweep approach to the algorithm
    self.policyparameter = policyparameter  # policy evaluation parameter
    self.environment = environment  # environment class
    self.agent = policy  # agent class
    self.verbose = verbose
    # Get the number of all possible states depending on Environment Type
    if isinstance(self.environment.observation_space, spaces.MultiDiscrete):
        self.value_func = np.zeros(self.environment.observation_space.nvec)
        self.state_type = 'MultiDiscrete'
    if isinstance(self.environment.observation_space, spaces.Discrete):
        self.value_func = np.zeros(self.environment.observation_space.n)
        self.state_type = 'Discrete'
    else:
        raise NotImplementedError("Only MultiDiscrete and Discrete are currently supported")
```

The parameter `policy` contains a class which initializes a decision rule depending on the Markov decision problem.

***

The algorithm policy iteration can be easily decomposed into two components, the so-called evaluation step and the improvement step. These steps are outsourced to internal helper methods `self._evaluate` and `self.improve`. The algorithm is executed until the decision rule converges with respect to some metric. In our case, the supremum norm is used as a criterion for changing the decision rule.

```python
def policy_iteration(self) -> None:
    """ using policy iteration to find optimal policy """
    done = False
    while not done:
        old_policy = self.agent.policy
        self._evaluate()
        self._improve()
        if (np.abs(self.agent.policy - old_policy) < self.policyparameter.epsilon).all():
            done = True
```

***

Next, we take a closer look at the components of the algorithm in detail. We start with the policy improvement step.
Here, the current state action function is used to create the new greedy policy.
Internally, the value of `self.agent.policy` changes, which is improved using the state action function from `q_values`.
The mapping `self._calculate_q_function_general` calculates the state action function based on the current value function from `self.value_func`.

```python
def _improve(self) -> None:
    """ improve the current policy of the agent by using a policy improvement step
    """
    # improve policy of the agent
    q_values = self._calculate_q_function_general()
    max_indices = np.argmax(q_values, axis=-1)
    self.agent.policy = np.zeros_like(self.agent.policy)
    self.agent.policy[tuple(np.indices(
        q_values.shape[:-1])) + (max_indices,)] = 1.0
```

***

Next, we consider the `_evaluate` method.
The mapping computes the associated value function for the current decision rule from `self.agent.policy` using policy evaluation.

This happens as follows:

```python
def _evaluate(self) -> None:
    """
    use policy evaluation to get the value function from the current policy"""
    # Update the value function according the current policy
    value_func_new = self.value_func.copy()
    done = False
    while not done:
        # store the new value function in the value_func_new variable
        q_values = self._calculate_q_function_general()
        if self.state_type == "Discrete":
            states = range(self.environment.observation_space.n)
        elif self.state_type == "MultiDiscrete":
            states = [tuple(state) for state in np.ndindex(self.value_func.shape)]
        for state in states:
            new_value = 0.0
            for action in range(self.environment.action_space.n):
                new_value += self.agent.policy[state][action] * \
                    q_values[state][action]
            value_func_new[state] = new_value
        # check if the new value function is in an epsilon environment of the old value function
        if (np.abs(value_func_new - self.value_func) < self.policyparameter.epsilon).all():
            done = True
            return
        self.value_func = value_func_new.copy()
```

For each state and action the state action function is updated using the rule from the lecture. These updates happen until the change of the value function is smaller than $\varepsilon$.

***

The last point is the calculation of the state action function. This is done using the following method:

```python
def _calculate_q_function_general(self) -> np.ndarray:
    q_values = np.zeros_like(self.agent.policy)  # initialize q_values
    # Update weighted Value

    for state in np.ndindex(self.value_func.shape):
        # in discrete case we only need integer values
        if len(state) == 1:
            state = int(state[0])
        # Get and Play all possible actions
        valid_actions = self.environment.get_valid_actions(state)
        for act in valid_actions:
            prob_next_states = self.environment.calculate_probability(
                state=state, action=act)
            rewards = self.environment.get_rewards(
                state=state, action=act)

            reward_state_action = 0.0
            value_function_next_step = 0.0
            if self.state_type == 'Discrete':
                for i, prob_next_state in enumerate(prob_next_states):
                    reward_state_action += prob_next_state * rewards[i]
                for i, prob_next_state in enumerate(prob_next_states):
                    value_function_next_step += prob_next_state * \
                        self.value_func[i]
            else:
                reward_state_action = np.sum(prob_next_states * rewards)
                value_function_next_step = np.sum(
                    prob_next_states * self.value_func)

            q_values[state][act] += reward_state_action + \
                self.policyparameter.gamma*value_function_next_step
    return q_values
```

The idea here is to determine the objects using the transition probability and the reward function as well as the rule for the theoretical relationship between the state action function and the value function.
