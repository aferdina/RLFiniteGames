""" use montecarlo approach to optimize grid world example
"""
from rlfinitegames.environments.grid_world import GridWorld
from rlfinitegames.algorithms.first_visit import MonteCarloPolicyIteration
from rlfinitegames.policies.discrete_agents import FiniteAgent
from rlfinitegames.algorithms.helperclasses import PolicyIterationParameter, MonteCarloPolicyIterationParameters, MonteCarloApproaches

SIZE = 5
TOTAL_STEPS = 10


def main():
    game = GridWorld(size=SIZE)
    agent = FiniteAgent(env=game)
    monte_carlo_parameters = MonteCarloPolicyIterationParameters(
        montecarloapproach="StateActionFunction", stateactionfunctioninit=20.0, invalidstateactionvalue=-1000000.0)
    policy_parameter = PolicyIterationParameter(
        epsilon_greedy=0.3, gamma=0.95, approach="Naive", decay_steps=3, epsilon=0.01, epsilon_greedy_decay=0.95)
    # run policy iteration
    algo = MonteCarloPolicyIteration(environment=game, policy=agent,
                                     montecarloparameter=monte_carlo_parameters, policyparameter=policy_parameter)
    # algo.policy_iteration()
    algo.policy_iteration_monte_carlo()
    game.reset()

    for _ in range(TOTAL_STEPS):
        action = algo.agent.get_action(game.state)
        print(f"action is {action}")
        _next_state, reward, done, _ = game.step(action)
        print(f"next state: {_next_state}, reward: {reward}, done: {done}")
        print(done)
        if done:
            game.reset()
        game.render()


if __name__ == '__main__':
    main()
