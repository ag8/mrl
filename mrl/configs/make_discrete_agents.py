from mrl.algorithms.discrete_off_policy import RandomPolicy, DistributionalQNetwork
from mrl.import_all import *
from argparse import Namespace
import gym
import time

from mrl.modules.train import DoNothing


def make_dqn_agent(base_config=default_dqn_config,
                   args=Namespace(env='InvertedPendulum-v2',
                                  tb='',
                                  parent_folder='/tmp/mrl',
                                  layers=(256, 256),
                                  num_envs=None),
                   agent_name_attrs=['env', 'seed', 'tb'],
                   **kwargs):
    if callable(base_config):  # If the base_config parameter is a function, make sure to call it
        base_config = base_config()
    config = base_config

    # Train on as many environments as the CPU allows,
    # unless specified otherwise.
    if hasattr(args, 'num_envs') and args.num_envs is None:
        import multiprocessing as mp
        args.num_envs = max(mp.cpu_count() - 1, 1)

    # set the prefix (todo: why?)
    if not hasattr(args, 'prefix'):
        args.prefix = 'dqn'

    # set whatever this is (todo: why?)
    if not args.tb:
        args.tb = str(time.time())

    merge_args_into_config(args, config)

    config.agent_name = make_agent_name(config, agent_name_attrs, prefix=args.prefix)

    base_modules = {
        k: v
        for k, v in dict(module_train=StandardTrain(),
                         module_eval=EpisodicEval(),
                         module_policy=QValuePolicy(),
                         module_logger=Logger(),
                         module_state_normalizer=Normalizer(MeanStdNormalizer()),
                         module_replay=OnlineHERBuffer(),
                         module_action_noise=None,
                         module_algorithm=DQN()).items() if not k in config
    }

    config.update(base_modules)

    if type(args.env) is str:
        env = lambda: gym.make(args.env)
        eval_env = env
    else:
        env = args.env
        eval_env = env

    if hasattr(args, 'eval_env') and args.eval_env is not None:
        if type(args.eval_env) is str:
            eval_env = lambda: gym.make(args.eval_env)
        else:
            eval_env = args.eval_env

    config.module_train_env = EnvModule(env, num_envs=config.num_envs, seed=config.seed)
    config.module_eval_env = EnvModule(eval_env, num_envs=config.num_eval_envs, name='eval_env',
                                       seed=config.seed + 1138)

    layer_norm = nn.LayerNorm if (hasattr(args, 'layer_norm') and args.layer_norm) else nn.Identity

    e = config.module_eval_env
    config.module_qvalue = PytorchModel(
        'qvalue',
        lambda: Actor(FCBody(e.state_dim + e.goal_dim, args.layers, layer_norm, make_activ(config.activ)), e.action_dim,
                      e.max_action))

    if e.goal_env:
        config.never_done = True  # important for standard Gym goal environments, which are never done

    return config


def make_distributionaldqn_agent(base_config=default_dqn_config,
                   args=Namespace(env='InvertedPendulum-v2',
                                  tb='',
                                  parent_folder='/tmp/mrl',
                                  layers=(256, 256),
                                  num_envs=None),
                   agent_name_attrs=['env', 'seed', 'tb'],
                   **kwargs):
    if callable(base_config):  # If the base_config parameter is a function, make sure to call it
        base_config = base_config()
    config = base_config

    # Train on as many environments as the CPU allows,
    # unless specified otherwise.
    if hasattr(args, 'num_envs') and args.num_envs is None:
        import multiprocessing as mp
        args.num_envs = max(mp.cpu_count() - 1, 1)

    # set the prefix (todo: why?)
    if not hasattr(args, 'prefix'):
        args.prefix = 'dqn'

    # set whatever this is (todo: why?)
    if not args.tb:
        args.tb = str(time.time())

    merge_args_into_config(args, config)

    config.agent_name = make_agent_name(config, agent_name_attrs, prefix=args.prefix)

    base_modules = {
        k: v
        for k, v in dict(module_train=StandardTrain(),
                         module_eval=EpisodicEval(),
                         module_policy=QValuePolicy(),
                         module_logger=Logger(),
                         module_state_normalizer=Normalizer(MeanStdNormalizer()),
                         module_replay=OnlineHERBuffer(),
                         module_action_noise=None,
                         module_algorithm=DistributionalQNetwork(num_atoms=5, v_min=-10, v_max=20)).items() if not k in config
    }

    config.update(base_modules)

    if type(args.env) is str:
        env = lambda: gym.make(args.env)
        eval_env = env
    else:
        env = args.env
        eval_env = env

    if hasattr(args, 'eval_env') and args.eval_env is not None:
        if type(args.eval_env) is str:
            eval_env = lambda: gym.make(args.eval_env)
        else:
            eval_env = args.eval_env

    config.module_train_env = EnvModule(env, num_envs=config.num_envs, seed=config.seed)
    config.module_eval_env = EnvModule(eval_env, num_envs=config.num_eval_envs, name='eval_env',
                                       seed=config.seed + 1138)

    layer_norm = nn.LayerNorm if (hasattr(args, 'layer_norm') and args.layer_norm) else nn.Identity

    e = config.module_eval_env
    config.module_qvalue = PytorchModel(
        'qvalue',
        lambda: Actor(FCBody(e.state_dim + e.goal_dim, args.layers, layer_norm, make_activ(config.activ)), e.action_dim,
                      e.max_action))

    if e.goal_env:
        config.never_done = True  # important for standard Gym goal environments, which are never done

    return config


def make_random_agent(base_config=default_ddpg_config,
                      args=Namespace(env='InvertedPendulum-v2',
                                     tb='',
                                     parent_folder='/tmp/mrl',
                                     layers=(256, 256),
                                     num_envs=None),
                      agent_name_attrs=['env', 'seed', 'tb'],
                      **kwargs):
    if callable(base_config):  # If the base_config parameter is a function, make sure to call it
        base_config = base_config()
    config = base_config

    # Train on as many environments as the CPU allows,
    # unless specified otherwise.
    if hasattr(args, 'num_envs') and args.num_envs is None:
        import multiprocessing as mp
        args.num_envs = max(mp.cpu_count() - 1, 1)

    # set the prefix (todo: why?)
    if not hasattr(args, 'prefix'):
        args.prefix = 'random'

    # set whatever this is (todo: why?)
    if not args.tb:
        args.tb = str(time.time())

    merge_args_into_config(args, config)

    config.agent_name = make_agent_name(config, agent_name_attrs, prefix=args.prefix)

    base_modules = {
        k: v
        for k, v in dict(module_train=DoNothing(),
                         module_eval=EpisodicEval(),
                         module_policy=RandomPolicy(),
                         module_logger=Logger(),
                         module_state_normalizer=None,
                         module_replay=None,
                         module_action_noise=None,
                         module_algorithm=None).items() if not k in config
    }

    config.update(base_modules)

    if type(args.env) is str:
        env = lambda: gym.make(args.env)
        eval_env = env
    else:
        env = args.env
        eval_env = env

    if hasattr(args, 'eval_env') and args.eval_env is not None:
        if type(args.eval_env) is str:
            eval_env = lambda: gym.make(args.eval_env)
        else:
            eval_env = args.eval_env

    config.module_train_env = EnvModule(env, num_envs=config.num_envs, seed=config.seed)
    config.module_eval_env = EnvModule(eval_env, num_envs=config.num_eval_envs, name='eval_env',
                                       seed=config.seed + 1138)

    layer_norm = nn.LayerNorm if (hasattr(args, 'layer_norm') and args.layer_norm) else nn.Identity

    e = config.module_eval_env
    config.module_actor = PytorchModel(
        'actor',
        lambda: Actor(FCBody(e.state_dim + e.goal_dim, args.layers, layer_norm, make_activ(config.activ)), e.action_dim,
                      e.max_action))
    config.module_critic = PytorchModel(
        'critic', lambda: Critic(
            FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, layer_norm, make_activ(config.activ)), 1))

    if e.goal_env:
        config.never_done = True  # important for standard Gym goal environments, which are never done

    return config
