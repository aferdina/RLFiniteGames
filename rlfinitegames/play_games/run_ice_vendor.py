"""run policy iteration on ice vendor game"""
from rlfinitegames.policies.discrete_agents import FiniteAgent
from rlfinitegames.environments.ice_vendor import IceVendor, GameConfig
from rlfinitegames.algorithms.policy_iteration import PolicyIteration, PolicyIterationParameter

TOTALSTEPS = 10
GAME_CONFIG = GameConfig(
    demand_parameters={"lam": 2.0}, storage_cost=1.0, selling_price=5.0)
POLICY_ITERATION_PARAMS = PolicyIterationParameter(
    approach='Naive', epsilon=0.01, gamma=0.9)


def main():
    """use policy iteration on ice vendor game"""
    env_ice = IceVendor(game_config=GAME_CONFIG)

    agent = FiniteAgent(env=env_ice)

    algo = PolicyIteration(environment=env_ice,
                           policy=agent, policyparameter=POLICY_ITERATION_PARAMS)
    algo.policy_iteration()
    env_ice.reset()

    for _ in range(TOTALSTEPS):
        action = algo.agent.get_action(env_ice.state)
        print(f"action: {action}")
        _next_state, _reward, done, _ = env_ice.step(action)
        env_ice.render()
        if done:
            env_ice.reset()


if __name__ == "__main__":
    main()
