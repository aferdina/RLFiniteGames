"""run policy iteration on ice vendor game"""
from rlfinitegames.policies.discrete_agents import FiniteAgent
from rlfinitegames.environments.ice_vendor import IceVendor, GameConfig
from rlfinitegames.environments.grid_world import GridWorld
from rlfinitegames.algorithms.one_step_dp import OneStepDynamicProgrammingParameters, OneStepDynamicProgramming, OneStepDynamicProgrammingInitConfig
import logging
from rlfinitegames.logging_module.setup_logger import setup_logger

# statics for logging purposes
LOGGINGPATH = "rlfinitegames/logging_module/logfiles/"
FILENAME = "run_sarsa_policy_iteration"
LOGGING_LEVEL = logging.INFO
LOGGING_CONSOLE_LEVEL = logging.INFO
LOGGING_FILE_LEVEL = logging.INFO

# statics for game purposes
TOTALSTEPS = 10
GAME_CONFIG = GameConfig(
    demand_parameters={"lam": 4.0}, storage_cost=1.0, selling_price=5.0)
ONES_STEP_DP_PARAMETERS = OneStepDynamicProgrammingParameters(
    epsilon=0.01, gamma=0.99, epsilon_greedy=0.4, epsilon_greedy_decay=0.95, decay_steps=5)
ONE_STEP_DP_INIT_CONFIG = OneStepDynamicProgrammingInitConfig(
    stateactionfunctioninit=20.0, invalidstateactionvalue=-100000)


def main():
    """use policy iteration on ice vendor game"""

    logger = setup_logger(logger_name=__name__,
                          logger_level=LOGGING_LEVEL, log_file=LOGGINGPATH + FILENAME + ".log",
                          file_handler_level=LOGGING_FILE_LEVEL, stream_handler_level=LOGGING_CONSOLE_LEVEL, console_output=True)
    ### ICE_VENDOR_GAME###
    # env_ice = IceVendor(game_config=GAME_CONFIG)
    # agent = FiniteAgent(env=env_ice)

    ### GRID_WORLD###
    logger.info("Initializing GRID_WORLD")
    env_grid = GridWorld(size=5)
    logger.info("Initializing Agent")
    agent = FiniteAgent(env=env_grid, policy_type="uniform")
    logger.info("Start algorithm")
    algo = OneStepDynamicProgramming(environment=env_grid,
                                     policy=agent, policyparameter=ONES_STEP_DP_PARAMETERS, init_parameter=ONE_STEP_DP_INIT_CONFIG)
    algo.policy_iteration_sarsa_evalulation()
    env_grid.reset()
    logger.info("run trained agent")
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

    for _ in range(TOTALSTEPS):
        action = algo.agent.get_action(env_grid.state)
        logger.info(f"action: {action}")
        _next_state, _reward, done, _ = env_grid.step(action)
        env_grid.render()
        if done:
            env_grid.reset()


if __name__ == "__main__":
    main()