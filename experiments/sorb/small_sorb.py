from envs.goalgridworld.goal_grid import GoalGridWorldEnv
from mrl.configs.make_discrete_agents import *


def make_env():
    grid_file = '2_room_9x9.txt'
    random_init_loc = False
    env = GoalGridWorldEnv(grid_size=5, max_step=25, grid_file=grid_file, random_init_loc=random_init_loc)

    return env


def test_distributionaldqn():
    print("About to test the DDQN!")

    print("Making agent...")
    config = make_ggw_test_agent(args=Namespace(
                                                # env='CartPole-v1',
                                                env=make_env,
                                                parent_folder='/tmp/mrl',
                                                layers=(64, 64,),
                                                max_episode_steps=1000,
                                                batch_size=17,
                                                warm_up=1000,
                                                initial_explore=1000,
                                                num_envs=1,
                                                num_eval_envs=1,
                                                use_distributional_rl=True,
                                                )
                                 )

    # config = make_sorbiquet_agent(args=Namespace(env=make_env,
    #                                              tb='',
    #                                              batch_size=256,
    #                                              target_network_update_frac=1.,
    #                                              target_network_update_freq=500,
    #                                              warm_up=2500,
    #                                              optimize_every=1,
    #                                              gamma=0.98,
    #                                              activ='relu',
    #                                              parent_folder='/tmp/mrl',
    #                                              layers=(64, 64,),
    #                                              max_episode_steps=1000,
    #                                              ensemble_size=3,
    #                                              use_distributional_rl=True,
    #                                              num_envs=1,
    #                                              num_atoms=5,
    #                                              v_min=-1,
    #                                              v_max=1,
    #                                              device='cpu'))

    agent = mrl.config_to_agent(config)
    print("Made agent successfully!")

    print("Training agent...")
    agent.train(num_steps=4000, render=False)
    agent.train(num_steps=100, render=True)
    # assert len(agent.eval(num_episodes=1).rewards) == 1
    # agent.train(num_steps=10000, render=True)
    # print("Trained agent...")
    # agent.eval(num_episodes=10)


test_distributionaldqn()
