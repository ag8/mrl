from mrl.configs.make_discrete_agents import *
import gym_minigrid


def test_dqn():
    print("About to test the DQN!")

    print("Making agent...")
    config = make_dqn_agent(args=Namespace(env='LunarLander-v2',
                                           tb='',
                                           batch_size=256,
                                           target_network_update_frac=1.,
                                           target_network_update_freq=500,
                                           warm_up=2500,
                                           optimize_every=1,
                                           gamma=0.98,
                                           activ='relu',
                                           parent_folder='/tmp/mrl',
                                           layers=(64,64,),
                                           num_envs=1,
                                           device='cpu'))

    agent = mrl.config_to_agent(config)
    print("Made agent successfully!")

    print("Training agent...")
    agent.train(num_steps=50000)
    assert len(agent.eval(num_episodes=1).rewards) == 1
    agent.train(num_steps=10000, render=True)
    print("Trained agent...")
    agent.eval(num_episodes=10)


test_dqn()
