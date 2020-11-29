from envs.goalgridworld.goal_grid import GoalGridWorldEnv
from mrl.configs.make_discrete_agents import *


def goalgridworld_env(grid_file='room_3x3_empty.txt'):
    """
    This is a custom environment function that creates the GoalGridWorld environment.

    :param grid_file: what maze to load. (default: effusion-midwalled 9x9 room)
    :return: the gym.GoalEnv environment
    """
    random_init_loc = False
    random_goal_loc = False
    env = GoalGridWorldEnv(grid_size=5, max_step=25, grid_file=grid_file, random_init_loc=random_init_loc,
                           random_goal_loc=random_goal_loc)

    return env


def test_sorb():
    print("Running SoRB!")

    max_episode_steps = 20

    print("Making agent...")
    config = get_distance_test_agent_config(args=Namespace(
        env=goalgridworld_env,  # load the GoalGridWorld environment
        parent_folder='/tmp/mrl',  # where to store the logs
        layers=(74,),  # shape of net
        max_episode_steps=max_episode_steps,  # maximum steps per episode
        batch_size=2,  # the batch size
        warm_up=500,  # how many steps to take randomly in order to fill up the replay buffer
        initial_explore=500,  # bugbug how is this distinct from the previous parameter?
        num_envs=1,  # number of training environments
        num_eval_envs=1,  # number of testing environments
        use_distributional_rl=True,  # whether to use distributional RL (if false, it will just use the clipping trick)
        target_network_update_freq=5,
        log_every=1000,
        qvalue_lr=1e-4,
        optimize_every=1,
        gamma=1

    )
    )

    agent = mrl.config_to_agent(config)
    print("Made agent successfully!")

    t = time.time()
    print("Training agent...")
    agent.train(num_steps=300000, render=False)

    # png_dir = '.'
    # images = []
    # for file_name in sorted(os.listdir(png_dir)):
    #     if file_name.endswith('.png'):
    #         file_path = os.path.join(png_dir, file_name)
    #         images.append(imageio.imread(file_path))
    #         os.remove(file_path)
    # imageio.mimsave('movie.gif', images, fps=2)

    # agent.train(num_steps=30, render=True)
    # assert len(agent.eval(num_episodes=1).rewards) == 1
    # agent.train(num_steps=10000, render=True)
    # print("Trained agent...")
    # agent.eval(num_episodes=10)
    print("Trained successfully.")
    elapsed = time.time() - t
    print("Elapsed time: " + str(elapsed) + "s.")


test_sorb()
