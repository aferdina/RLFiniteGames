from rlfinitegames.environments.grid_world import GridWorld
from rlfinitegames.algorithms.double_state_action_learning import DoubleParameter
from rlfinitegames.environments.constructed_max_bias import ConstructedMaxBias
from rlfinitegames.algorithms.policy_iteration import FiniteAgent
from rlfinitegames.algorithms.helperclasses import RunTimeMethod, UpdateMethod, TruncatedBounds
from rlfinitegames.algorithms.double_state_action_learning import DoubleStateActionLearning, DoubleParameter


def main():
    TOTALSTEPS = 10
    SIZE = 5
    parameter = DoubleParameter()
    env = GridWorld(size=SIZE)
    agent = FiniteAgent(env)
    algo = DoubleStateActionLearning(
        environment=env, policy=agent, algo_params=parameter)
    algo.run_double_state_action_learning()
    env.reset()
    ### Constructed Max Bias ###

    # env = ConstructedMaxBias(number_arms=5)
    # agent = FiniteAgent(env=env, policy_type="uniform")
    # parameter = DoubleParameter(
    #     epsilon=0.01, runtimemethod=RunTimeMethod.CRITERION.value, episodes=10000, updatemethod=UpdateMethod.CLIPPED.value, trunc_bounds=TruncatedBounds(lower_bound=10.0, upper_bound=10.0), method=PolicyMethod.EPSILONGREEDY.value)
    # algo = DoubleStateActionLearning(
    #     environment=env, policy=agent, algo_params=parameter)
    # algo.run_double_state_action_learning()
    # env.reset()
    for _ in range(TOTALSTEPS):
        action = algo.agent.get_action(env.state)
        print(f"action: {action}")
        _next_state, _reward, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()


if __name__ == "__main__":
    main()
