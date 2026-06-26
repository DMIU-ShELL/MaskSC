#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

'''
lifelong (continual) learning experiments using supermask
superpostion algorithm in RL.
https://arxiv.org/abs/2006.14769
'''

import json
import copy
import shutil
import matplotlib
matplotlib.use("Pdf")
from deep_rl import *
import os
import argparse

##### ContinualWorld environment
'''
sac, baseline (no lifelong learning), task boundary (oracle) given
'''
def sac_baseline_continualworld(name, args):
    env_config_path = args.env_config_path
    task_label_input_disabled = args.disable_task_label_input

    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 3e-4
    config.cl_preservation = 'baseline'
    config.seed = args.seed
    random_seed(config.seed)
    exp_suffix = '-no_task_label' if task_label_input_disabled else ''
    exp_id = '-{0}{1}'.format(config.seed, exp_suffix)
    log_name = name + '-sac' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 1

    # get num_tasks from env_config
    with open(env_config_path, 'r') as f:
        env_config_ = json.load(f)
    num_tasks = len(env_config_['tasks'])
    del env_config_
    config.use_task_label_input = not task_label_input_disabled

    task_fn = lambda log_dir: ContinualWorld(name, env_config_path, log_dir, config.seed)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    eval_task_fn = lambda log_dir: ContinualWorld(name, env_config_path, log_dir, config.seed)
    config.eval_task_fn = eval_task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.Adam(params, lr=lr)
    config.actor_optimizer_fn = lambda params, lr: torch.optim.Adam(params, lr=lr)
    config.critic_optimizer_fn = lambda params, lr: torch.optim.Adam(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: SACTanhGaussianNet_CL(
        state_dim, action_dim, label_dim,
        phi_body=DummyBody_CL(
            state_dim,
            task_label_dim=None if task_label_input_disabled else label_dim),
        actor_body=LayerNormFCBody_CL(
            state_dim + (0 if task_label_input_disabled else label_dim),
            hidden_units=(256, 256, 256, 256)),
        critic_body=LayerNormFCBody_CL(
            state_dim + (0 if task_label_input_disabled else label_dim) + action_dim,
            hidden_units=(256, 256, 256, 256)))
    config.state_normalizer = RunningStatsNormalizer()
    config.reward_normalizer = RewardRunningStatsNormalizer()
    config.discount = 0.99
    config.rollout_length = 500
    config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.sac_batch_size = 256
    config.sac_replay_size = int(1e6)
    config.sac_tau = 5e-3
    config.sac_updates_per_step = 1
    config.sac_init_random_steps = 1000
    config.sac_min_replay_size = 1000
    config.sac_alpha = 0.2
    config.sac_auto_entropy_tuning = True
    config.sac_alpha_tuning = 'target_std'
    config.sac_target_std = 0.089
    config.max_steps = args.max_steps
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir, file_name='train-log')
    config.cl_requires_task_label = True
    config.log_parameter_histograms = args.log_parameter_histograms
    config.histogram_log_interval = args.histogram_log_interval
    config.save_task_checkpoints = args.save_task_checkpoints

    config.eval_interval = 200
    config.task_ids = np.arange(num_tasks).tolist()

    agent = SACBaselineAgent(config)
    config.agent_name = agent.__class__.__name__
    tasks = agent.config.cl_tasks_info
    config.cl_num_learn_blocks = 1
    shutil.copy(env_config_path, config.log_dir + '/env_config.json')
    with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    run_iterations_w_oracle(agent, tasks)
    with open('{0}/tasks_info_after_train.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    # save config
    with open('{0}/config.json'.format(config.log_dir), 'w') as f:
        dict_config = vars(config)
        for k in dict_config.keys():
            if not isinstance(dict_config[k], int) \
            and not isinstance(dict_config[k], float) and dict_config[k] is not None:
                dict_config[k] = str(dict_config[k])
        json.dump(dict_config, f)
'''
sac, supermask lifelong learning, task boundary (oracle) given
'''
def sac_ll_continualworld(name, args):
    env_config_path = args.env_config_path
    task_label_input_disabled = args.disable_task_label_input

    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 3e-4
    config.cl_preservation = 'supermask'
    config.seed = args.seed
    random_seed(config.seed)
    exp_suffix = '-no_task_label' if task_label_input_disabled else ''
    exp_id = '-{0}-mask-{1}{2}'.format(config.seed, args.new_task_mask, exp_suffix)
    log_name = args.pathheader + '/' + name + '-sac' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 1

    # get num_tasks from env_config
    with open(env_config_path, 'r') as f:
        env_config_ = json.load(f)
    num_tasks = len(env_config_['tasks'])
    del env_config_
    config.use_task_label_input = not task_label_input_disabled

    task_fn = lambda log_dir: ContinualWorld(name, env_config_path, log_dir, config.seed)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    eval_task_fn = lambda log_dir: ContinualWorld(name, env_config_path, log_dir, config.seed)
    config.eval_task_fn = eval_task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.Adam(params, lr=lr)
    config.actor_optimizer_fn = lambda params, lr: torch.optim.Adam(params, lr=lr)
    config.critic_optimizer_fn = lambda params, lr: torch.optim.Adam(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: SACTanhGaussianNet_SS(
        state_dim, action_dim, label_dim,
        phi_body=DummyBody_CL(
            state_dim,
            task_label_dim=None if task_label_input_disabled else label_dim),
        actor_body=LayerNormFCBody_SS(
            state_dim + (0 if task_label_input_disabled else label_dim),
            hidden_units=(256, 256, 256, 256),
            discrete_mask=False, num_tasks=num_tasks, new_task_mask=args.new_task_mask),
        critic_body=LayerNormFCBody_SS(
            state_dim + (0 if task_label_input_disabled else label_dim) + action_dim,
            hidden_units=(256, 256, 256, 256), discrete_mask=False,
            num_tasks=num_tasks, new_task_mask=args.new_task_mask),
        num_tasks=num_tasks, new_task_mask=args.new_task_mask)
    config.state_normalizer = RunningStatsNormalizer()
    config.reward_normalizer = RewardRunningStatsNormalizer()
    config.discount = 0.99
    config.rollout_length = 50
    config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.sac_batch_size = 256
    config.sac_replay_size = int(1e6)
    config.sac_tau = 5e-3
    config.sac_updates_per_step = 1
    config.sac_init_random_steps = 10_000 #1000     # in env steps
    config.sac_min_replay_size = 1000               # in env steps
    config.sac_alpha = 0.01
    config.sac_auto_entropy_tuning = True
    config.sac_alpha_tuning = 'target_std'  # auto_entropy, target_std
    config.sac_target_std = 0.089
    config.max_steps = args.max_steps
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir, file_name='train-log')
    config.cl_requires_task_label = True
    config.log_parameter_histograms = args.log_parameter_histograms
    config.histogram_log_interval = args.histogram_log_interval
    config.save_task_checkpoints = args.save_task_checkpoints

    config.eval_interval = 2000
    config.task_ids = np.arange(num_tasks).tolist()

    config.detect_reference_num = 50
    config.detect_num_samples = 1000
    config.detect_frequency = 10
    config.legacy_wte_ema = args.legacy_wte_ema
    config.detect_fn = lambda input_dim, action_dim: Detect(config.detect_reference_num,
        input_dim, action_dim, config.detect_num_samples, one_hot=False, normalized=True)
    config.detect_topk = None
    config.select_frequency = 20 # was 5, 20 * rollout length of 50 = 1000 env steps == warm up phase.
    config.select_strategy = args.select_strategy
    config.select_once_per_task = args.select_once_per_task

    ###
    # iteration = rollout length * workers = 50 * 1 = 50 env steps
    # sac warm up = sac_min_replay_size = 1000 env steps
    # sac_init_random_steps = 10_000 env steps
    # SAR vector is ~ 39 + 4 + 1 = 44 dimensions (state, action, reward) for continual world
    # embedding construction = 50
    # indices selection =
    # evaluation block = 400 * (50 rollout length) = 10_000 steps (for 1_000_000 max steps 50 evaluations)
    # updates =


    agent = SACDetectLLAgent(config)
    config.agent_name = agent.__class__.__name__
    tasks = agent.config.cl_tasks_info
    config.cl_num_learn_blocks = 1
    shutil.copy(env_config_path, config.log_dir + '/env_config.json')
    with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    run_iterations_w_oracle(agent, tasks)
    with open('{0}/tasks_info_after_train.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    # save config
    with open('{0}/config.json'.format(config.log_dir), 'w') as f:
        dict_config = vars(config)
        for k in dict_config.keys():
            if not isinstance(dict_config[k], int) \
            and not isinstance(dict_config[k], float) and dict_config[k] is not None:
                dict_config[k] = str(dict_config[k])
        json.dump(dict_config, f)

if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    select_device(0) # -1 is CPU, a positive integer is the index of GPU

    parser = argparse.ArgumentParser()
    parser.add_argument('algo', help='algorithm to run')
    parser.add_argument('--env_name', help='name of the evaluation environment. ' \
        'minigrid and ctgraph currently supported', default='continualworld')
    parser.add_argument('--env_config_path', help='path to environment config', \
        default='./env_configs/continualworld_10.json')
    parser.add_argument('--max_steps', help='maximum number of training steps per task.', \
        default=10_240_000, type=int)
    parser.add_argument('--new_task_mask', help='', \
        default='random', type=str)
    parser.add_argument(
        '--legacy_wte_ema',
        '--legacy-wte-ema',
        dest='legacy_wte_ema',
        help=(
            'use the legacy WTE update, which averages the raw new embedding '
            'with the stored unit embedding before normalisation'
        ),
        action='store_true',
    )
    parser.add_argument('--select_strategy', help='selection strategy: similarity or random_topk', \
        default='similarity', choices=['similarity', 'random_topk'])
    parser.add_argument('--select_once_per_task', help='only run the first eligible selection per task', \
        action='store_true')
    parser.add_argument('--disable_task_label_input',
        help='do not concatenate the task label to the policy network input; task labels are still used for task switching/evaluation',
        action='store_true')
    parser.add_argument('--log_parameter_histograms',
        help='enable TensorBoard parameter histograms; disabled by default because they create very large event files',
        action='store_true')
    parser.add_argument('--histogram_log_interval',
        help='iteration interval for parameter histograms when --log_parameter_histograms is enabled; defaults to iteration_log_interval',
        type=int,
        default=1)
    parser.add_argument('--save_task_checkpoints',
        help='save full per-task model checkpoints under task_stats; disabled by default because these files are very large',
        action='store_true')
    parser.add_argument('--seed', help='seed for the experiment', default=8379, type=int)
    parser.add_argument('--pathheader', '--p', '-p', help='experiment header to log path for launcher.py', type=str, default='')
    args = parser.parse_args()

    if args.env_name == 'continualworld':
        name = Config.ENV_CONTINUALWORLD
        if args.algo == 'baseline':
            sac_baseline_continualworld(name, args)
        elif args.algo == 'll_supermask':
            sac_ll_continualworld(name, args)
        else:
            raise ValueError('algo {0} not implemented'.format(args.algo))
    else:
        raise ValueError('--env_name {0} not implemented'.format(args.env_name))
