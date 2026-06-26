import numpy as np
import pickle
import os
import time
import datetime
import torch
from .torch_utils import *
from ..mask_modules import set_selected_task_indices

_PRIOR_SELECTION_STRATEGIES = frozenset({
    'similarity',
    'random_topk',
    'oracle_all',
})

def _csv_float(value):
    if value is None:
        return 'nan'
    try:
        if not np.isfinite(value):
            return 'nan'
    except TypeError:
        return 'nan'
    return f'{float(value):.6f}'

def _selection_perf_eligible(cfg, prior_idx, current_perf, prior_own_perf):
    prior_perf = prior_own_perf[prior_idx]
    if not np.isfinite(prior_perf):
        return False

    min_perf = getattr(cfg, 'selection_prior_min_perf', None)
    if min_perf is not None and prior_perf < min_perf:
        return False

    if getattr(cfg, 'selection_require_prior_better_than_current', False):
        margin = getattr(cfg, 'selection_prior_margin', 0.0)
        if not np.isfinite(current_perf):
            current_perf = 0.0
        if prior_perf <= current_perf + margin:
            return False

    return True

def _filter_selected_priors(cfg, selected, task_idx, latest_task_perf, prior_own_perf):
    current_perf = latest_task_perf[task_idx]
    return [
        prior_idx for prior_idx in selected
        if _selection_perf_eligible(cfg, prior_idx, current_perf, prior_own_perf)
    ]

def _select_oracle_all(task_idx, family_stride):
    family_stride = int(family_stride)
    if family_stride <= 0:
        raise ValueError('family_stride must be a positive integer')
    return [
        prior_idx for prior_idx in range(task_idx)
        if prior_idx % family_stride == task_idx % family_stride
    ]

def _reset_selected_priors_for_new_task(agent):
    config = agent.config
    select_strategy = getattr(config, 'select_strategy', None)
    select_frequency = getattr(config, 'select_frequency', 0) or 0
    if select_strategy not in _PRIOR_SELECTION_STRATEGIES or select_frequency <= 0:
        return False

    # A selector starts each task using only the current task mask. This
    # prevents the selected set from the preceding task leaking into the
    # warm-up period before the first selection event.
    set_selected_task_indices(agent.network, [])
    return True

def _should_log_parameter_histograms(config, iteration):
    if not getattr(config, 'log_parameter_histograms', False):
        return False
    interval = getattr(config, 'histogram_log_interval', None)
    if interval is None:
        interval = getattr(config, 'iteration_log_interval', 1)
    interval = int(interval)
    return interval > 0 and iteration % interval == 0

def _should_save_iteration_snapshots(config, iteration):
    if not getattr(config, 'save_iteration_snapshots', False):
        return False
    interval = getattr(config, 'iteration_snapshot_interval', None)
    if interval is None:
        interval = getattr(config, 'iteration_log_interval', 1)
    interval = int(interval)
    return interval > 0 and iteration % interval == 0

def _itr_log(logger, agent, iteration, dict_logs):
    logger.info('iteration %d, total steps %d, mean/max/min reward %f/%f/%f'%(
        iteration, agent.total_steps,
        np.mean(agent.iteration_rewards),
        np.max(agent.iteration_rewards),
        np.min(agent.iteration_rewards)
    ))
    logger.scalar_summary('last_episode_reward/avg', np.mean(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/std', np.std(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/max', np.max(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/min', np.min(agent.last_episode_rewards))
    logger.scalar_summary('iteration_reward/avg', np.mean(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/std', np.std(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/max', np.max(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/min', np.min(agent.iteration_rewards))

    if hasattr(agent, 'layers_output'):
        for tag, value in agent.layers_output:
            value = value.detach().cpu().numpy()
            value_norm = np.linalg.norm(value, axis=-1)
            logger.scalar_summary('debug/{0}_avg_norm'.format(tag), np.mean(value_norm))
            logger.scalar_summary('debug/{0}_avg'.format(tag), value.mean())
            logger.scalar_summary('debug/{0}_std'.format(tag), value.std())
            logger.scalar_summary('debug/{0}_max'.format(tag), value.max())
            logger.scalar_summary('debug/{0}_min'.format(tag), value.min())

    for key, value in dict_logs.items():
        logger.scalar_summary('debug_extended/{0}_avg'.format(key), np.mean(value))
        logger.scalar_summary('debug_extended/{0}_std'.format(key), np.std(value))
        logger.scalar_summary('debug_extended/{0}_max'.format(key), np.max(value))
        logger.scalar_summary('debug_extended/{0}_min'.format(key), np.min(value))

    return

# metaworld/continualworld
def _itr_log_mw(logger, agent, iteration, dict_logs):
    logger.info('iteration %d, total steps %d, mean/max/min reward %f/%f/%f, ' \
        'mean/max/min success rate %f/%f/%f'%(
        iteration, agent.total_steps,
        np.mean(agent.iteration_rewards),
        np.max(agent.iteration_rewards),
        np.min(agent.iteration_rewards),
        np.mean(agent.iteration_success_rate),
        np.max(agent.iteration_success_rate),
        np.min(agent.iteration_success_rate)
    ))
    logger.scalar_summary('last_episode_reward/avg', np.mean(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/std', np.std(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/max', np.max(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/min', np.min(agent.last_episode_rewards))
    logger.scalar_summary('iteration_reward/avg', np.mean(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/std', np.std(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/max', np.max(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/min', np.min(agent.iteration_rewards))

    logger.scalar_summary('last_episode_success_rate/avg', np.mean(agent.last_episode_success_rate))
    logger.scalar_summary('last_episode_success_rate/std', np.std(agent.last_episode_success_rate))
    logger.scalar_summary('last_episode_success_rate/max', np.max(agent.last_episode_success_rate))
    logger.scalar_summary('last_episode_success_rate/min', np.min(agent.last_episode_success_rate))
    logger.scalar_summary('iteration_success_rate/avg', np.mean(agent.iteration_success_rate))
    logger.scalar_summary('iteration_success_rate/std', np.std(agent.iteration_success_rate))
    logger.scalar_summary('iteration_success_rate/max', np.max(agent.iteration_success_rate))
    logger.scalar_summary('iteration_success_rate/min', np.min(agent.iteration_success_rate))

    if hasattr(agent, 'layers_output'):
        for tag, value in agent.layers_output:
            value = value.detach().cpu().numpy()
            value_norm = np.linalg.norm(value, axis=-1)
            logger.scalar_summary('debug/{0}_avg_norm'.format(tag), np.mean(value_norm))
            logger.scalar_summary('debug/{0}_avg'.format(tag), value.mean())
            logger.scalar_summary('debug/{0}_std'.format(tag), value.std())
            logger.scalar_summary('debug/{0}_max'.format(tag), value.max())
            logger.scalar_summary('debug/{0}_min'.format(tag), value.min())

    for key, value in dict_logs.items():
        logger.scalar_summary('debug_extended/{0}_avg'.format(key), np.mean(value))
        logger.scalar_summary('debug_extended/{0}_std'.format(key), np.std(value))
        logger.scalar_summary('debug_extended/{0}_max'.format(key), np.max(value))
        logger.scalar_summary('debug_extended/{0}_min'.format(key), np.min(value))

    return

# run iterations, lifelong learning
# used by either a baseline agent (with no task knowledge preservation) or
# an agent with knowledge preservation via supermask superposition (ss)
# modules on: PPO agent or PPO agent with supermask
# modules off: detect and resource manager
def run_iterations_w_oracle(agent, tasks_info):
    config = agent.config

    log_path_tstats = config.log_dir + '/task_stats'
    if not os.path.exists(log_path_tstats):
        os.makedirs(log_path_tstats)
    log_path_eval = config.log_dir + '/eval_stats'
    if not os.path.exists(log_path_eval):
        os.makedirs(log_path_eval)
    random_seed(config.seed)
    agent_name = agent.__class__.__name__

    iteration = 0
    steps = []
    rewards = []
    task_start_idx = 0
    num_tasks = len(tasks_info)
    # track how many times each task selected each prior task (for post-run summaries)
    selection_counts = [[0 for _ in range(num_tasks)] for _ in range(num_tasks)]
    prior_own_perf = np.full(num_tasks, np.nan, dtype=np.float32)
    latest_task_perf = np.full(num_tasks, np.nan, dtype=np.float32)
    best_task_perf = np.full(num_tasks, -np.inf, dtype=np.float32)
    eval_data_fh = open(config.logger.log_dir + '/eval_metrics.csv', 'a', buffering=1)
    sims_csv_path = config.logger.log_dir + '/task_similarities.csv'
    sims_csv_fh = open(sims_csv_path, 'w', buffering=1)
    sims_csv_fh.write(
        'learn_block,task_idx,iteration,total_steps,prev_idx,similarity,'
        'selected,prior_perf,current_perf,eligible\n'
    )

    eval_tracker = False
    eval_data = []
    metric_icr = [] # icr => total cumulative reward

    if agent.task.name == config.ENV_METAWORLD or agent.task.name == config.ENV_CONTINUALWORLD:
        itr_log_fn = _itr_log_mw
    else:
        itr_log_fn = _itr_log

    for learn_block_idx in range(config.cl_num_learn_blocks):
        config.logger.info('********** start of learning block {0}'.format(learn_block_idx))
        eval_results = {task_idx:[] for task_idx in range(len(tasks_info))}

        for task_idx, task_info in enumerate(tasks_info):
            config.logger.info('*****start training on task {0}'.format(task_idx))
            config.logger.info('name: {0}'.format(task_info['name']))
            config.logger.info('task: {0}'.format(task_info['task']))
            config.logger.info('task_label: {0}'.format(task_info['task_label']))

            states = agent.task.reset_task(task_info)
            agent.states = config.state_normalizer(states)
            agent.data_buffer.clear()
            agent.task_train_start(task_info['task_label'])
            if _reset_selected_priors_for_new_task(agent):
                config.logger.info(
                    f'Reset selected priors for task {task_idx}; '
                    'warm-up uses only the current task mask'
                )
            selected_once_for_task = False

            COS_TH = config.COS_TH

            while True:
                # ---- agent iteration ----
                dict_logs = agent.iteration()
                iteration += 1

                total_steps = agent.total_steps
                steps.append(total_steps)
                rewards.append(float(np.mean(agent.iteration_rewards)))

                # ---- locals to reduce attribute overhead
                cfg = agent.config
                select_strategy = getattr(cfg, "select_strategy", "similarity")
                detect_freq = getattr(cfg, "detect_frequency", 0) or 0
                select_freq = getattr(cfg, "select_frequency", 0) or 0
                detect_topk = getattr(cfg, "detect_topk", None)
                select_once_per_task = getattr(cfg, "select_once_per_task", False)

                # ---- detect / embedding update ----
                if hasattr(agent, "detect") and select_strategy == "similarity":
                    if iteration != 0 and iteration % agent.config.detect_frequency == 0 and agent.data_buffer.size() >= (agent.detect.get_num_samples()):
                        # extract SAR batch of 128 samples
                        sar_data = agent.extract_sar(batch_size=config.detect_num_samples)

                        # Update
                        new_embedding = agent.compute_task_embedding(sar_data, agent.task.action_dim)
                        agent._update_embedding(task_idx=task_idx, new_emb=new_embedding, ema=0.5)

                # ---- selection step ----
                should_select = (
                    iteration
                    and select_freq
                    and (iteration % select_freq == 0)
                    and task_idx > 0
                    and (not select_once_per_task or not selected_once_for_task)
                )
                if should_select:
                    selection_attempted = False
                    if select_strategy == "oracle_all":
                        family_stride = getattr(cfg, "family_stride", None)
                        if family_stride is None:
                            raise ValueError(
                                "select_strategy='oracle_all' requires config.family_stride"
                            )
                        selected = _select_oracle_all(task_idx, family_stride)
                        selection_attempted = True

                        # Oracle-All means every available same-family predecessor:
                        # no similarity threshold, top-k cap, or competence filter.
                        set_selected_task_indices(agent.network, selected)
                        for idx in selected:
                            selection_counts[task_idx][idx] += 1

                        selected_set = set(selected)
                        lines = [
                            (
                                f"{learn_block_idx},{task_idx},{iteration},{total_steps},"
                                f"{prev_idx},nan,{int(prev_idx in selected_set)},"
                                f"{_csv_float(prior_own_perf[prev_idx])},"
                                f"{_csv_float(latest_task_perf[task_idx])},"
                                f"{int(prev_idx in selected_set)}\n"
                            )
                            for prev_idx in range(task_idx)
                        ]
                        sims_csv_fh.writelines(lines)
                        cfg.logger.info(
                            f"Oracle-All priors (family_stride={family_stride}): {selected}"
                        )

                    elif select_strategy == "random_topk":
                        candidate_indices = list(range(task_idx))
                        k = min(getattr(cfg, "detect_topk", 0) or 0, task_idx)
                        selected_raw = np.random.choice(candidate_indices, size=k, replace=False).tolist() if k > 0 else []
                        selected = _filter_selected_priors(
                            cfg, selected_raw, task_idx, latest_task_perf, prior_own_perf
                        )
                        selection_attempted = True

                        set_selected_task_indices(agent.network, selected)
                        for idx in selected:
                            selection_counts[task_idx][idx] += 1

                        selected_set = set(selected)
                        # buffer writes
                        lines = [
                            (
                                f"{learn_block_idx},{task_idx},{iteration},{total_steps},"
                                f"{prev_idx},nan,{int(prev_idx in selected_set)},"
                                f"{_csv_float(prior_own_perf[prev_idx])},"
                                f"{_csv_float(latest_task_perf[task_idx])},"
                                f"{int(_selection_perf_eligible(cfg, prev_idx, latest_task_perf[task_idx], prior_own_perf))}\n"
                            )
                            for prev_idx in range(task_idx)
                        ]
                        sims_csv_fh.writelines(lines)
                        cfg.logger.info(f"Random priors: {selected_raw}\nFiltered selected: {selected}")

                    elif select_strategy == "similarity":
                        # select prior indices
                        selected_raw, sims = agent.select_similar(task_idx=task_idx, threshold=COS_TH, topk=detect_topk)
                        selection_attempted = sims is not None
                        selected = _filter_selected_priors(
                            cfg, selected_raw, task_idx, latest_task_perf, prior_own_perf
                        )

                        if selection_attempted:
                            set_selected_task_indices(agent.network, selected)
                        for idx in selected:
                            selection_counts[task_idx][idx] += 1

                        selected_set = set(selected)

                        # log sims for *existing* embeddings (fast + consistent with cache)
                        # sims is aligned with agent._emb_indices
                        lines = []
                        sims_list = []
                        if sims is not None:
                            sims_cpu = sims.detach().float().cpu().tolist()
                            for sim_val, prev_idx in zip(sims_cpu, agent._emb_indices):
                                sims_list.append((sim_val, prev_idx))
                                eligible = _selection_perf_eligible(
                                    cfg, prev_idx, latest_task_perf[task_idx], prior_own_perf
                                )
                                lines.append(
                                    (
                                        f"{learn_block_idx},{task_idx},{iteration},{total_steps},"
                                        f"{prev_idx},{sim_val:.6f},{int(prev_idx in selected_set)},"
                                        f"{_csv_float(prior_own_perf[prev_idx])},"
                                        f"{_csv_float(latest_task_perf[task_idx])},"
                                        f"{int(eligible)}\n"
                                    )
                                )
                            sims_csv_fh.writelines(lines)
                            sims_list.sort(key=lambda x: x[0], reverse=True)

                        cfg.logger.info(f"Prior sims: {sims_list}\nRaw selected: {selected_raw}\nFiltered selected: {selected}")

                    if select_once_per_task and selection_attempted:
                        selected_once_for_task = True
                                
                # ---- logging iteration stats ----
                if iteration % config.iteration_log_interval == 0:
                    itr_log_fn(config.logger, agent, iteration, dict_logs)

                    if _should_save_iteration_snapshots(config, iteration):
                        with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % \
                            (agent_name, config.tag, agent.task.name), 'wb') as f:
                            pickle.dump({'rewards': rewards, 'steps': steps}, f)
                        agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_name, config.tag, \
                            agent.task.name))
                    if _should_log_parameter_histograms(config, iteration):
                        for tag, value in agent.network.named_parameters():
                            tag = tag.replace('.', '/')
                            config.logger.histo_summary(tag, value.data.cpu().numpy())
                        if hasattr(agent, 'layers_output'):
                            for tag, value in agent.layers_output:
                                tag = 'layer_output/' + tag
                                config.logger.histo_summary(tag, value.data.cpu().numpy())

                # ---- evaluation block ----
                if (agent.config.eval_interval is not None and \
                    iteration % agent.config.eval_interval == 0):
                    config.logger.info('*****agent / evaluation block')
                    _tasks = tasks_info
                    _names = [eval_task_info['name'] for eval_task_info in _tasks]
                    config.logger.info('eval tasks: {0}'.format(', '.join(_names)))
                    eval_data.append(np.zeros(len(_tasks),))
                    for eval_task_idx, eval_task_info in enumerate(_tasks):
                        agent.task_eval_start(eval_task_info['task_label'])
                        eval_states = agent.evaluation_env.reset_task(eval_task_info)
                        agent.evaluation_states = eval_states
                        # performance (perf) can be success rate in (meta-)continualworld or
                        # rewards in other environments
                        perf, eps = agent.evaluate_cl(num_iterations=config.evaluation_episodes)
                        agent.task_eval_end()
                        mean_perf = float(np.mean(perf))
                        eval_data[-1][eval_task_idx] = mean_perf
                        latest_task_perf[eval_task_idx] = mean_perf
                        best_task_perf[eval_task_idx] = max(best_task_perf[eval_task_idx], mean_perf)
                    _record = np.concatenate([eval_data[-1], np.array(time.time()).reshape(1,)])
                    np.savetxt(eval_data_fh, _record.reshape(1, -1), delimiter=',', fmt='%.4f')
                    del _record
                    icr = eval_data[-1].sum()
                    metric_icr.append(icr)
                    tpot = np.sum(metric_icr)
                    config.logger.info('*****cl evaluation:')
                    config.logger.info('cl eval ICR: {0}'.format(icr))
                    config.logger.info('cl eval TPOT: {0}'.format(tpot))
                    config.logger.scalar_summary('cl_eval/icr', icr)
                    config.logger.scalar_summary('cl_eval/tpot', np.sum(metric_icr))


                # check whether task training has been completed
                task_steps_limit = config.max_steps * (num_tasks * learn_block_idx + task_idx + 1)
                if config.max_steps and agent.total_steps >= task_steps_limit:
                    with open(log_path_tstats + '/%s-%s-online-stats-%s-run-%d-task-%d.bin' % \
                        (agent_name, config.tag, agent.task.name, learn_block_idx+1, task_idx+1), 'wb') as f:
                        pickle.dump({'rewards': rewards[task_start_idx : ], \
                        'steps': steps[task_start_idx : ]}, f)

                    if hasattr(agent, 'seen_tasks'):
                        config.logger.info('cacheing mask for current task')
                    ret = agent.task_train_end()
                    if getattr(config, 'save_task_checkpoints', False):
                        agent.save(log_path_tstats +'/%s-%s-model-%s-run-%d-task-%d.bin' % (agent_name, \
                            config.tag, agent.task.name, learn_block_idx+1, task_idx+1))
                    agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_name, config.tag, \
                        agent.task.name))
                    task_start_idx = len(rewards)
                    break
            # end of while True. current task training
            # evaluate agent across task exposed to agent so far
            config.logger.info('evaluating agent across all tasks exposed so far to agent')
            for j in range(task_idx+1):
                _eval_task = tasks_info[j]
                agent.task_eval_start(_eval_task['task_label'])

                eval_states = agent.evaluation_env.reset_task(tasks_info[j])
                agent.evaluation_states = eval_states
                perf, episodes = agent.evaluate_cl(num_iterations=config.evaluation_episodes)
                eval_results[j] += perf
                mean_perf = float(np.mean(perf))
                latest_task_perf[j] = mean_perf
                best_task_perf[j] = max(best_task_perf[j], mean_perf)
                if j == task_idx:
                    prior_own_perf[task_idx] = mean_perf
                    config.logger.info(
                        'stored prior own performance for task {0}: {1:.4f}'.format(
                            task_idx, mean_perf
                        )
                    )

                agent.task_eval_end()

                with open(log_path_eval+'/rewards-task{0}_{1}.bin'.format(\
                    task_idx+1, j+1), 'wb') as f:
                    pickle.dump(perf, f)
                with open(log_path_eval+'/episodes-task{0}_{1}.bin'.format(\
                    task_idx+1, j+1), 'wb') as f:
                    pickle.dump(episodes, f)
        # end for each task
        print('eval stats')
        with open(log_path_eval + '/eval_full_stats.bin', 'wb') as f: pickle.dump(eval_results, f)

        f = open(log_path_eval + '/eval_stats.csv', 'w')
        f.write('task_id,avg_reward\n')
        for k, v in eval_results.items():
            print('{0}: {1:.4f}'.format(k, np.mean(v)))
            f.write('{0},{1:.4f}\n'.format(k, np.mean(v)))
            config.logger.scalar_summary('zeval/task_{0}/avg_reward'.format(k), np.mean(v))
        f.close()
        config.logger.info('********** end of learning block {0}\n'.format(learn_block_idx))
    # end for learning block
    eval_data_fh.close()
    sims_csv_fh.close()
    if hasattr(agent, 'detect'):
        config.logger.info('***** selection counts (current task -> prior task: count)')
        print('selection counts (current task -> prior task: count)')
        for curr_idx in range(num_tasks):
            if curr_idx == 0:
                summary = 'none'
            else:
                summary = ', '.join([f'{prior}:{selection_counts[curr_idx][prior]}' for prior in range(curr_idx)])
            config.logger.info(f'task {curr_idx}: {summary}')

    if len(eval_data) > 0:
        to_save = np.stack(eval_data, axis=0)
        with open(config.logger.log_dir + '/eval_metrics.npy', 'wb') as f:
            np.save(f, to_save)
    agent.close()
    return steps, rewards
