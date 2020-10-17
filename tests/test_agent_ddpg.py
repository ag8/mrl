from mrl.configs.make_continuous_agents import *
import numpy as np
import pytest


def test_ddpg():
    print("About to test the DDPG!")

    print("Making agent...")
    config = make_ddpg_agent(args=Namespace(env='BipedalWalker-v3',
                                            tb='',
                                            parent_folder='/tmp/mrl',
                                            layers=(32, 1),
                                            num_envs=1,
                                            device='cpu'))
    agent = mrl.config_to_agent(config)
    print("Made agent successfully!")

    print("Training agent...")
    agent.train(num_steps=100000)
    agent.train(num_steps=10000, render=True)
    assert len(agent.eval(num_episodes=1).rewards) == 1
    print("Trained agent...")


test_ddpg()
