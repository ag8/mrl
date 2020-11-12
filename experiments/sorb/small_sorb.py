from envs.goalgridworld.goal_grid import GoalGridWorldEnv
from mrl.configs.make_discrete_agents import *


def make_env(grid_file='2_room_9x9.txt'):
    """
    This is a custom environment function that creates the GoalGridWorld environment.

    :param grid_file: what maze to load. (default: effusion-midwalled 9x9 room)
    :return: the gym.GoalEnv environment
    """
    random_init_loc = False
    env = GoalGridWorldEnv(grid_size=5, max_step=25, grid_file=grid_file, random_init_loc=random_init_loc)

    return env


def test_distributionaldqn():
    print("Running SoRB!")

    print("Making agent...")
    config = make_ggw_test_agent(args=Namespace(
        env=make_env,  # load the GoalGridWorld environment
        parent_folder='/tmp/mrl',  # where to store the logs
        layers=(64, 64,),  # shape of... bugbug what?
        max_episode_steps=1000,  # maximum steps per episode
        batch_size=1,  # the batch size
        warm_up=3000,  # how many steps to take randomly in order to fill up the replay buffer
        initial_explore=3000,  # bugbug how is this distinct from the previous parameter?
        num_envs=1,  # number of training environments
        num_eval_envs=1,  # number of testing environments
        use_distributional_rl=True,  # whether to use distributional RL (if false, it will just use the clipping trick)
    )
    )

    agent = mrl.config_to_agent(config)
    print("Made agent successfully!")

    t = time.time()
    print("Training agent...")
    agent.train(num_steps=4000, render=False)
    agent.train(num_steps=100, render=True)
    # assert len(agent.eval(num_episodes=1).rewards) == 1
    # agent.train(num_steps=10000, render=True)
    # print("Trained agent...")
    # agent.eval(num_episodes=10)
    print("Trained successfully.")
    elapsed = time.time() - t
    print("Elapsed time: " + str(elapsed) + "ms.")


test_distributionaldqn()
