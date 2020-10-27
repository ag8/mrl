import time
from argparse import Namespace

import gym

from mrl.algorithms.discrete_off_policy import DistributionalQNetwork, SearchPolicy, SorbDDQN
from mrl.import_all import *


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
        lambda: Critic(FCBody(e.state_dim, args.layers, layer_norm, make_activ(config.activ)), e.action_dim)
    )

    if e.goal_env:
        config.never_done = True  # important for standard Gym goal environments, which are never done

    return config


def make_distributionaldqn_agent(base_config=default_dqn_config,
                                 args=Namespace(env='InvertedPendulum-v2',
                                                tb='',
                                                parent_folder='/tmp/mrl',
                                                layers=(256, 256),
                                                num_envs=None,
                                                num_atoms=5,
                                                v_min=-1,
                                                v_max=1),
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
                         module_algorithm=DistributionalQNetwork(num_atoms=args.num_atoms, v_min=args.v_min,
                                                                 v_max=args.v_max)).items()
        if
        not k in config
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
        lambda: Critic(FCBody(e.state_dim, args.layers, layer_norm, make_activ(config.activ)),
                       e.action_dim * args.num_atoms)
    )

    if e.goal_env:
        config.never_done = True  # important for standard Gym goal environments, which are never done

    return config


def make_sorbiquet_agent(base_config=default_dqn_config,
                         args=Namespace(env='InvertedPendulum-v2',
                                        tb='',
                                        parent_folder='/tmp/mrl',
                                        layers=(256, 256),
                                        num_envs=None,
                                        num_atoms=5,
                                        v_min=-1,
                                        v_max=1,
                                        max_episode_steps=1000,
                                        ensemble_size=3,
                                        combine_ensemble_method='min',
                                        use_distributional_rl=True),
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
        args.prefix = 'sorb_ddqn_agent'

    # set whatever this is (todo: why?)
    if not args.tb:
        args.tb = str(time.time())

    merge_args_into_config(args, config)
    config.env = args.env  # for some reason this doesn't happen automatically with merge_args

    config.agent_name = make_agent_name(config, agent_name_attrs, prefix=args.prefix)

    base_modules = {
        k: v
        for k, v in dict(module_train=StandardTrain(),
                         module_eval=EpisodicEval(),
                         module_policy=SearchPolicy(),
                         module_logger=Logger(),
                         module_state_normalizer=Normalizer(MeanStdNormalizer()),
                         module_replay=OnlineHERBuffer(),
                         module_action_noise=None,
                         module_algorithm=SorbDDQN(num_atoms=5, v_max=1, v_min=-1)).items()
        if k not in config
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
        lambda: Critic(FCBody(e.state_dim, args.layers, layer_norm, make_activ(config.activ)),
                       e.action_dim * args.num_atoms)
    )

    assert args.max_episode_steps is not None
    config._max_episode_steps = args.max_episode_steps
    config._ensemble_size = args.ensemble_size
    config._use_distributional_rl = args.use_distributional_rl

    e = config.module_eval_env

    if e.goal_env:
        config.never_done = True  # important for standard Gym goal environments, which are never done

    return config


def make_ggw_test_agent(base_config=default_dqn_config,
                        args=Namespace(env='InvertedPendulum-v2',
                                       tb='',
                                       parent_folder='/tmp/mrl',
                                       layers=(256, 256),
                                       num_envs=None,
                                       num_atoms=5,
                                       v_min=-1,
                                       v_max=1,
                                       max_episode_steps=1000,
                                       ensemble_size=3,
                                       combine_ensemble_method='min',
                                       use_distributional_rl=True),
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
        args.prefix = 'sorb_ddqn_agent'

    # set whatever this is (todo: why?)
    if not hasattr(args, 'tb'):
        args.tb = str(time.time())
    if not args.tb:
        args.tb = str(time.time())

    merge_args_into_config(args, config)
    config.env = args.env  # for some reason this doesn't happen automatically with merge_args

    config.agent_name = make_agent_name(config, agent_name_attrs, prefix=args.prefix)

    base_modules = {
        k: v
        for k, v in dict(module_train=StandardTrain(),
                         module_eval=EpisodicEval(),
                         module_policy=SearchPolicy(),
                         module_logger=Logger(),
                         module_state_normalizer=Normalizer(MeanStdNormalizer()),
                         module_replay=OnlineHERBuffer(),
                         module_action_noise=None,
                         module_algorithm=SorbDDQN(num_atoms=5, v_max=1, v_min=-1)).items()
        if k not in config
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
        lambda: Critic(FCBody(e.state_dim * 2,  # * 2 because of stacked goal
                              args.layers, layer_norm, make_activ(config.activ)),
                       e.action_dim)
    )

    # assert args.max_episode_steps is not None
    # config._max_episode_steps = args.max_episode_steps
    # config._ensemble_size = args.ensemble_size
    # config._use_distributional_rl = args.use_distributional_rl

    e = config.module_eval_env

    if e.goal_env:
        config.never_done = True  # important for standard Gym goal environments, which are never done

    return config
