from mrl.configs.make_discrete_agents import *


def test_dqn():
    print("About to test the DQN!")

    print("Making agent...")
    config = make_dqn_agent(args=Namespace(env='CartPole-v1',
                                           tb='',
                                           parent_folder='/tmp/mrl',
                                           layers=(32, 1),
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
