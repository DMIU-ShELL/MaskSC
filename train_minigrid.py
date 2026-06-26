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

##### Minigrid environment
'''
ppo, baseline (no lifelong learning), task boundary (oracle) given
'''
def ppo_baseline_minigrid(name, args):
    env_config_path = args.env_config_path
    task_label_input_disabled = args.disable_task_label_input

    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.cl_preservation = 'baseline'
    config.seed = args.seed
    random_seed(config.seed)
    exp_suffix = '-no_task_label' if task_label_input_disabled else ''
    exp_id = '-{0}{1}'.format(config.seed, exp_suffix)
    log_name = name + '-ppo' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 4

    # get num_tasks from env_config
    with open(env_config_path, 'r') as f:
        env_config_ = json.load(f)
    num_tasks = len(env_config_['tasks'])
    del env_config_
    config.cl_num_tasks = num_tasks
    config.use_task_label_input = not task_label_input_disabled

    task_fn = lambda log_dir: MiniGridFlatObs(name, env_config_path, log_dir, config.seed, False)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    eval_task_fn = lambda log_dir: MiniGridFlatObs(name, env_config_path, log_dir, config.seed, True)
    config.eval_task_fn = eval_task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_CL(
        state_dim, action_dim, label_dim, 
        phi_body=FCBody_CL(
            state_dim,
            task_label_dim=None if task_label_input_disabled else label_dim,
            hidden_units=(200, 200, 200),
        ),
        actor_body=DummyBody_CL(200),
        critic_body=DummyBody_CL(200))
    config.policy_fn = SamplePolicy
    #config.state_normalizer = ImageNormalizer()
    # rescale state normaliser: suitable for grid encoding of states in minigrid
    config.state_normalizer = RescaleNormalizer(1./10.)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.1 #0.75
    config.rollout_length = 128
    config.optimization_epochs = 8
    config.num_mini_batches = 64
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.max_steps = args.max_steps
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir, file_name='train-log')
    config.cl_requires_task_label = True
    config.reset_optimizer_on_task_change = args.reset_optimizer_on_task_change
    config.log_parameter_histograms = args.log_parameter_histograms
    config.histogram_log_interval = args.histogram_log_interval
    config.save_task_checkpoints = args.save_task_checkpoints
    config.save_iteration_snapshots = args.save_iteration_snapshots
    config.iteration_snapshot_interval = args.iteration_snapshot_interval

    config.eval_interval = 20
    config.task_ids = np.arange(num_tasks).tolist()

    agent = BaselineAgent(config)
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
ppo, supermask lifelong learning, task boundary (oracle) given
'''
def ppo_ll_minigrid(name, args):
    env_config_path = args.env_config_path
    task_label_input_disabled = args.disable_task_label_input

    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.cl_preservation = 'supermask'
    config.seed = args.seed
    random_seed(config.seed)
    exp_suffix = '-no_task_label' if task_label_input_disabled else ''
    exp_id = '-{0}-mask-{1}{2}-{3}'.format(config.seed, args.new_task_mask, exp_suffix, args.exp_id)
    log_name = args.pathheader + '/' + name + '-ppo' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 4
    # get num_tasks from env_config
    with open(env_config_path, 'r') as f:
        env_config_ = json.load(f)
    num_tasks = len(env_config_['tasks'])
    del env_config_
    config.cl_num_tasks = num_tasks
    config.use_task_label_input = not task_label_input_disabled

    task_fn = lambda log_dir: MiniGridFlatObs(name, env_config_path, log_dir, config.seed, False)
    #task_fn = lambda log_dir: MiniGrid(name, env_config_path, log_dir, config.seed, False)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    eval_task_fn = lambda log_dir: MiniGridFlatObs(name, env_config_path, log_dir, config.seed, True)
    #eval_task_fn = lambda log_dir: MiniGrid(name, env_config_path, log_dir, config.seed, True)
    config.eval_task_fn = eval_task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(
        state_dim, action_dim, label_dim,
        phi_body=FCBody_SS(
            state_dim,
            task_label_dim=None if task_label_input_disabled else label_dim,
            hidden_units=(200, 200, 200),
            num_tasks=num_tasks,
            new_task_mask=args.new_task_mask,
        ),
        actor_body=DummyBody_CL(200),
        critic_body=DummyBody_CL(200),

        #phi_body=ConvBody_SS_Modified(
        #    state_dim,
        #    feature_dim=256,
        #    task_label_dim=None if task_label_input_disabled else label_dim,
        #    num_tasks=config.cl_num_tasks,
        #    new_task_mask=args.new_task_mask,
        #    seed=config.seed
        #    ),
        #actor_body=DummyBody_CL(256),  #200
        #critic_body=DummyBody_CL(256),

        num_tasks=num_tasks,
        new_task_mask=args.new_task_mask)
    config.policy_fn = SamplePolicy
    #config.state_normalizer = ImageNormalizer()
    # rescale state normaliser: suitable for grid encoding of states in minigrid
    config.state_normalizer = RescaleNormalizer(1./10.)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.01 #0.75
    config.rollout_length = 128
    config.optimization_epochs = 8
    config.num_mini_batches = 64
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.max_steps = args.max_steps
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir, file_name='train-log')
    config.cl_requires_task_label = True
    config.reset_optimizer_on_task_change = args.reset_optimizer_on_task_change
    config.log_parameter_histograms = args.log_parameter_histograms
    config.histogram_log_interval = args.histogram_log_interval
    config.save_task_checkpoints = args.save_task_checkpoints
    config.save_iteration_snapshots = args.save_iteration_snapshots
    config.iteration_snapshot_interval = args.iteration_snapshot_interval

    config.eval_interval = 20
    config.task_ids = np.arange(num_tasks).tolist()

    #=============================================================#
    #                   Mask-SC Hyperparameters
    #=============================================================#
    config.detect_reference_num = 50
    config.detect_num_samples = 512
    config.detect_frequency = 1
    config.legacy_wte_ema = args.legacy_wte_ema
    config.detect_fn = lambda input_dim, action_dim: Detect(
        config.detect_reference_num,
        input_dim, action_dim,
        config.detect_num_samples,
        one_hot=True,
        normalized=True
    )
    config.detect_topk = None  # Pick top 3 masks in pre-selection
    config.COS_TH = 0.65    # CT-graph 0.5, MiniGrid 0.75
    config.select_frequency = 1
    config.select_strategy = args.select_strategy
    config.select_once_per_task = args.select_once_per_task
    config.family_stride = args.family_stride

    config.selection_prior_min_perf = args.selection_prior_min_perf
    config.selection_require_prior_better_than_current = args.selection_require_prior_better_than_current
    config.selection_prior_margin = args.selection_prior_margin
    #=============================================================#

    agent = DetectLLAgent(config)
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
        'minigrid and ctgraph currently supported', default='minigrid')
    parser.add_argument('--env_config_path', help='path to environment config', \
        default='./env_configs/minigrid_10.json')
    parser.add_argument('--exp_id', help='experiment id', default='mg10', type=str)
    parser.add_argument('--max_steps', help='maximum number of training steps per task.', \
        default=51200*5, type=int)
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
    parser.add_argument(
        '--select_strategy',
        help='prior selection strategy',
        default='similarity',
        choices=['similarity', 'random_topk', 'oracle_all'],
    )
    parser.add_argument(
        '--family_stride',
        help=(
            'number of interleaved task families; Oracle-All treats tasks '
            'with equal task_idx modulo this value as one family'
        ),
        type=int,
        default=4,
    )

    parser.add_argument('--select_once_per_task', help='only run the first eligible selection per task', \
        action='store_true')
    parser.add_argument('--disable_task_label_input',
        help='do not concatenate the task label to the policy network input; task labels are still used for task switching/evaluation',
        action='store_true')
    parser.add_argument('--reset_optimizer_on_task_change',
        help='recreate the RMSprop optimizer at each task boundary',
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
    parser.add_argument('--save_iteration_snapshots',
        help='save latest model and online-stats snapshots during iteration logging; disabled by default because model snapshots are very large',
        action='store_true')
    parser.add_argument('--iteration_snapshot_interval',
        help='iteration interval for --save_iteration_snapshots; defaults to iteration_log_interval',
        type=int,
        default=None)
    parser.add_argument('--seed', help='seed for experiment', default=54741, type=int)
    parser.add_argument('--pathheader', '--p', '-p', help='experiment header to log path for launcher.py', type=str, default='')


    parser.add_argument(
        '--selection_prior_min_perf',
        help=(
            'minimum own-task evaluation performance required before a prior '
            'mask can be reused; omit to disable this competence gate'
        ),
        type=float,
        default=None,
    )
    parser.add_argument(
        '--selection_require_prior_better_than_current',
        help='also require prior own-task performance to exceed current task performance plus margin',
        action='store_true',
    )
    parser.add_argument(
        '--selection_prior_margin',
        help='margin used with --selection_require_prior_better_than_current',
        type=float,
        default=0.0,
    )
    args = parser.parse_args()

    if args.env_name == 'minigrid':
        name = Config.ENV_MINIGRID
        if args.algo == 'baseline':
            ppo_baseline_minigrid(name, args)
        elif args.algo == 'll_supermask':
            ppo_ll_minigrid(name, args)
        else:
            raise ValueError('algo {0} not implemented'.format(args.algo))
    else:
        raise ValueError('--env_name {0} not implemented'.format(args.env_name))
