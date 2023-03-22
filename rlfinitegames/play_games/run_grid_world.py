"""run policy iteration on grid world game"""
from rlfinitegames.policies.discrete_agents import FiniteAgent
from rlfinitegames.environments.grid_world import GridWorld
from rlfinitegames.algorithms.policy_iteration import PolicyIteration

TOTALSTEPS = 10


def main():
    """run policy iteration on grid world game"""
    # set size of grid world
    size = int(input("Please provide size of Grid World Environment: "))
    # initialize agent
    agent = FiniteAgent(env=GridWorld(size=size))
    # initialize grid world environment
    env = GridWorld(size=size)

    # run policy iteration
    algo = PolicyIteration(environment=env, policy=agent)
    algo.policy_iteration()
    env.reset()

    for _ in range(TOTALSTEPS):
        action = algo.agent.get_action(env.state)
        _next_state, reward, done, _ = env.step(action)
        print(f"next state: {_next_state}, reward: {reward}, done: {done}")
        print(done)
        if done:
            env.reset()
        env.render()
    print(
        f"Resulting policy: {algo.agent.policy}, Resulting Value Function: {algo.value_func}")


if __name__ == "__main__":
    main()
