#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from copy import deepcopy

from ..network import *
from ..component import *
from .BaseAgent import *
import numpy as np
from ..mask_modules import *


class SACContinualLearnerAgent(BaseContinualLearnerAgent):
    def __init__(self, config):
        BaseContinualLearnerAgent.__init__(self, config)
        self.config = config
        self.task = None if config.task_fn is None else config.task_fn()
        if config.eval_task_fn is None:
            self.evaluation_env = None
        else:
            self.evaluation_env = config.eval_task_fn(config.log_dir)
            self.task = self.evaluation_env if self.task is None else self.task

        tasks_ = self.task.get_all_tasks(config.cl_requires_task_label)
        tasks = [tasks_[task_id] for task_id in config.task_ids]
        del tasks_
        self.config.cl_tasks_info = tasks
        label_dim = 0 if tasks[0]['task_label'] is None else len(tasks[0]['task_label'])
        self.task_label_dim = label_dim

        torch.manual_seed(config.seed)
        self.network = self._build_network()
        self.target_network = deepcopy(self.network)
        self.target_network.load_state_dict(self.network.state_dict())

        self.actor_opt_fn = getattr(config, 'actor_optimizer_fn', None) or config.optimizer_fn
        self.critic_opt_fn = getattr(config, 'critic_optimizer_fn', None) or config.optimizer_fn
        self.actor_opt = self.actor_opt_fn(self._actor_parameters(), config.lr)
        self.critic_opt = self.critic_opt_fn(self._critic_parameters(), config.lr)

        alpha_tuning = getattr(config, 'sac_alpha_tuning', None)
        if alpha_tuning is None:
            if getattr(config, 'sac_auto_entropy_tuning', True):
                alpha_tuning = 'target_entropy'
            else:
                alpha_tuning = 'fixed'
        valid_alpha_tuning = {'fixed', 'target_entropy', 'target_std'}
        if alpha_tuning not in valid_alpha_tuning:
            raise ValueError(f'unknown SAC alpha tuning mode: {alpha_tuning}')
        self.alpha_tuning = alpha_tuning
        self.target_entropy = getattr(config, 'sac_target_entropy', -float(self.task.action_dim))
        self.target_std = float(getattr(config, 'sac_target_std', 0.089))
        self.auto_alpha = self.alpha_tuning != 'fixed'
        self.alpha = float(getattr(config, 'sac_alpha', 0.2))
        if self.auto_alpha:
            init_alpha = max(self.alpha, 1e-6)
            self.log_alpha = torch.tensor(np.log(init_alpha), device=Config.DEVICE,
                dtype=torch.float32, requires_grad=True)
            self.alpha_lr = getattr(config, 'sac_alpha_lr', config.lr)
            self.alpha_opt_fn = getattr(config, 'alpha_optimizer_fn', None)
            if self.alpha_opt_fn is None:
                self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
            else:
                self.alpha_opt = self.alpha_opt_fn([self.log_alpha], self.alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(max(self.alpha, 1e-6)), device=Config.DEVICE,
                dtype=torch.float32)
            self.alpha_opt = None
            self.alpha_lr = getattr(config, 'sac_alpha_lr', config.lr)
            self.alpha_opt_fn = getattr(config, 'alpha_optimizer_fn', None)

        self.total_steps = 0
        self.task_train_steps = 0
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)
        self.running_episodes_rewards = [[] for _ in range(config.num_workers)]
        self.iteration_rewards = np.zeros(config.num_workers)

        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.layers_output = None

        replay_size = int(getattr(config, 'sac_replay_size', 1e6))
        batch_size = int(getattr(config, 'sac_batch_size', 256))
        self.data_buffer = Replay(memory_size=replay_size, batch_size=batch_size)

        self.sac_batch_size = batch_size
        self.sac_tau = float(getattr(config, 'sac_tau', 5e-3))
        self.sac_updates_per_step = int(getattr(config, 'sac_updates_per_step', 1))
        self.sac_init_random_steps = int(getattr(config, 'sac_init_random_steps', 1000))
        self.sac_min_replay_size = int(getattr(config, 'sac_min_replay_size', batch_size))

        self.curr_train_task_label = None
        self.curr_eval_task_label = None

        if self.task.name == config.ENV_METAWORLD or self.task.name == config.ENV_CONTINUALWORLD:
            self._rollout_fn = self._rollout_metaworld
            self.episode_success_rate = np.zeros(config.num_workers)
            self.last_episode_success_rate = np.zeros(config.num_workers)
            self.running_episodes_success_rate = [[] for _ in range(config.num_workers)]
            self.iteration_success_rate = np.zeros(config.num_workers)
        else:
            self._rollout_fn = self._rollout_normal
            self.episode_success_rate = None
            self.last_episode_success_rate = None
            self.running_episodes_success_rate = None
            self.iteration_success_rate = None

    def _build_network(self):
        try:
            return self.config.network_fn(self.task.state_dim, self.task.action_dim,
                self.task_label_dim, self.task.action_space)
        except TypeError:
            return self.config.network_fn(self.task.state_dim, self.task.action_dim,
                self.task_label_dim)

    def _actor_parameters(self):
        params = []
        modules = [self.network.actor_body, self.network.fc_action, self.network.fc_log_std]
        for module in modules:
            params.extend([p for p in module.parameters() if p.requires_grad])
        return params

    def _critic_parameters(self):
        params = []
        modules = [self.network.phi_body, self.network.critic1_body, self.network.critic2_body,
            self.network.fc_q1, self.network.fc_q2]
        for module in modules:
            params.extend([p for p in module.parameters() if p.requires_grad])
        return params

    def save(self, filename):
        payload = {
            'network': self.network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'alpha': self.alpha,
        }
        torch.save(payload, filename)

    def load(self, filename):
        payload = torch.load(filename, map_location=lambda storage, loc: storage,
            weights_only=False)
        if isinstance(payload, dict) and 'network' in payload:
            self.network.load_state_dict(payload['network'])
            target_state = payload.get('target_network', payload['network'])
            self.target_network.load_state_dict(target_state)
            if 'log_alpha' in payload and hasattr(self, 'log_alpha'):
                self.log_alpha.data.copy_(payload['log_alpha'].to(self.log_alpha.device))
            self.alpha = float(payload.get('alpha', self.alpha))
            return

        self.network.load_state_dict(payload)
        self.target_network.load_state_dict(payload)

    def _repeat_task_label(self, batch_dim):
        if self.curr_train_task_label is not None:
            task_label = self.curr_train_task_label
        else:
            task_label = self.task.get_task()['task_label']
            assert False, 'manually set (temporary) breakpoint. code should not get here.'

        task_label = tensor(task_label)
        if batch_dim == 1:
            return task_label.reshape(1, -1)
        return torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

    def _sample_actions(self, states, batch_task_label):
        use_random = self.task_train_steps < self.sac_init_random_steps or \
            self.data_buffer.size() < self.sac_min_replay_size
        if use_random:
            actions = np.stack([self.task.action_space.sample() for _ in range(self.config.num_workers)])
            return actions, None

        with torch.no_grad():
            _, sampled_action, _, _, _, _ = self.network.predict(states, task_label=batch_task_label)
        return sampled_action.detach().cpu().numpy(), None

    def _update_episode_metrics(self, rewards, terminals):
        self.episode_rewards += rewards
        for i, terminal in enumerate(terminals):
            if terminal:
                self.running_episodes_rewards[i].append(self.episode_rewards[i])
                self.last_episode_rewards[i] = self.episode_rewards[i]
                self.episode_rewards[i] = 0

    def _finalize_iteration_rewards(self):
        for i in range(self.config.num_workers):
            self.iteration_rewards[i] = self._avg_episodic_perf(self.running_episodes_rewards[i])

    def _rollout_normal(self, states, batch_task_label):
        self.running_episodes_rewards = [[] for _ in range(self.config.num_workers)]

        for _ in range(self.config.rollout_length):
            actions, _ = self._sample_actions(states, batch_task_label)
            next_states, rewards, terminals, _ = self.task.step(actions)
            self._update_episode_metrics(rewards, terminals)
            rewards_norm = self.config.reward_normalizer(rewards)
            next_states = self.config.state_normalizer(next_states)
            self.data_buffer.feed_batch([states, actions, rewards_norm, terminals, next_states])
            states = next_states
            self.total_steps += self.config.num_workers
            self.task_train_steps += self.config.num_workers

        self._finalize_iteration_rewards()
        return states

    def _rollout_metaworld(self, states, batch_task_label):
        self.running_episodes_rewards = [[] for _ in range(self.config.num_workers)]
        self.running_episodes_success_rate = [[] for _ in range(self.config.num_workers)]

        for _ in range(self.config.rollout_length):
            actions, _ = self._sample_actions(states, batch_task_label)
            next_states, rewards, terminals, infos = self.task.step(actions)
            success_rates = [info['success'] for info in infos]
            self.episode_success_rate += success_rates
            self._update_episode_metrics(rewards, terminals)
            rewards_norm = self.config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminal:
                    self.episode_success_rate[i] = (self.episode_success_rate[i] > 0).astype(np.uint8)
                    self.running_episodes_success_rate[i].append(self.episode_success_rate[i])
                    self.last_episode_success_rate[i] = self.episode_success_rate[i]
                    self.episode_success_rate[i] = 0
            next_states = self.config.state_normalizer(next_states)
            self.data_buffer.feed_batch([states, actions, rewards_norm, terminals, next_states])
            states = next_states
            self.total_steps += self.config.num_workers
            self.task_train_steps += self.config.num_workers

        self._finalize_iteration_rewards()
        for i in range(self.config.num_workers):
            self.iteration_success_rate[i] = self._avg_episodic_perf(
                self.running_episodes_success_rate[i])
        return states

    def _avg_episodic_perf(self, running_perf):
        if len(running_perf) == 0:
            return 0.
        return np.mean(running_perf)

    def _set_modules_grad(self, modules, flag):
        for module in modules:
            for param in module.parameters():
                param.requires_grad_(flag)

    def _soft_update(self):
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.sac_tau) + param.data * self.sac_tau
            )

    def _reset_optimizers(self):
        self.actor_opt = self.actor_opt_fn(self._actor_parameters(), self.config.lr)
        self.critic_opt = self.critic_opt_fn(self._critic_parameters(), self.config.lr)
        if self.auto_alpha:
            if self.alpha_opt_fn is None:
                self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
            else:
                self.alpha_opt = self.alpha_opt_fn([self.log_alpha], self.alpha_lr)

    def _update_networks(self, batch_task_label):
        logs = {
            'critic_loss': [],
            'actor_loss': [],
            'alpha_loss': [],
            'q1': [],
            'q2': [],
            'target_q': [],
            'log_prob': [],
            'entropy': [],
            'alpha': [],
            'grad_norm': [],
        }

        if self.data_buffer.size() < max(self.sac_batch_size, self.sac_min_replay_size):
            self.layers_output = []
            return self._sanitize_logs(logs)

        critic_modules = [self.network.phi_body, self.network.critic1_body, self.network.critic2_body,
            self.network.fc_q1, self.network.fc_q2]
        actor_modules = [self.network.actor_body, self.network.fc_action, self.network.fc_log_std]
        update_steps = self.config.rollout_length * self.config.num_workers * self.sac_updates_per_step
        last_actor_outs = []

        for _ in range(update_steps):
            states, actions, rewards, terminals, next_states = self.data_buffer.sample(
                self.sac_batch_size)
            states = tensor(states)
            actions = tensor(actions)
            rewards = tensor(rewards).view(-1, 1)
            terminals = tensor(terminals).view(-1, 1)
            next_states = tensor(next_states)
            batch_dim = states.shape[0]
            batch_task_label = torch.repeat_interleave(batch_task_label[:1], batch_dim, dim=0)

            with torch.no_grad():
                next_actions, next_log_prob, _, _, _, _ = self.network.sample(
                    next_states,
                    task_label=batch_task_label,
                    deterministic=False,
                    return_layer_output=False,
                    with_logprob=True,
                )
                target_q1, target_q2, _ = self.target_network.q(
                    next_states, next_actions, task_label=batch_task_label)
                target_v = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
                q_target = rewards + self.config.discount * (1 - terminals) * target_v

            q1, q2, _ = self.network.q(states, actions, task_label=batch_task_label)
            critic_loss = 0.5 * ((q1 - q_target).pow(2).mean() + (q2 - q_target).pow(2).mean())

            self.critic_opt.zero_grad()
            critic_loss.backward()
            critic_grad = nn.utils.clip_grad_norm_(self._critic_parameters(),
                self.config.gradient_clip)
            self.critic_opt.step()

            self._set_modules_grad(critic_modules, False)
            policy_action, log_prob, entropy, _, policy_log_std, actor_outs = self.network.sample(
                states,
                task_label=batch_task_label,
                deterministic=False,
                return_layer_output=True,
                with_logprob=True,
            )
            q1_pi, q2_pi, q_layers = self.network.q(states, policy_action,
                task_label=batch_task_label, return_layer_output=True)
            actor_loss = (self.alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            actor_grad = nn.utils.clip_grad_norm_(self._actor_parameters(),
                self.config.gradient_clip)
            self.actor_opt.step()
            self._set_modules_grad(critic_modules, True)

            alpha_loss_value = torch.zeros(1, device=Config.DEVICE)
            if self.auto_alpha:
                if self.alpha_tuning == 'target_std':
                    avg_std = policy_log_std.exp().mean()
                    alpha_loss = (self.log_alpha * (avg_std - self.target_std).detach()).mean()
                elif self.alpha_tuning == 'auto_entropy':
                    alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()
                self.alpha = self.log_alpha.exp().detach().item()
                alpha_loss_value = alpha_loss.detach()

            self._soft_update()
            last_actor_outs = actor_outs + q_layers

            logs['critic_loss'].append(critic_loss.detach().cpu().item())
            logs['actor_loss'].append(actor_loss.detach().cpu().item())
            logs['alpha_loss'].append(alpha_loss_value.cpu().item())
            logs['q1'].append(q1.detach().cpu().mean().item())
            logs['q2'].append(q2.detach().cpu().mean().item())
            logs['target_q'].append(q_target.detach().cpu().mean().item())
            logs['log_prob'].append(log_prob.detach().cpu().mean().item())
            logs['entropy'].append(entropy.detach().cpu().mean().item())
            logs['alpha'].append(self.alpha)
            logs['grad_norm'].append(float(max(critic_grad.detach().cpu().item(),
                actor_grad.detach().cpu().item())))

        self.layers_output = last_actor_outs
        return self._sanitize_logs(logs)

    def _sanitize_logs(self, logs):
        for key, value in logs.items():
            if len(value) == 0:
                logs[key] = [0.0]
        return logs

    def iteration(self):
        batch_task_label = self._repeat_task_label(self.config.num_workers)
        self.states = self._rollout_fn(self.states, batch_task_label)
        return self._update_networks(batch_task_label)


class SACBaselineAgent(SACContinualLearnerAgent):
    def __init__(self, config):
        SACContinualLearnerAgent.__init__(self, config)

    def task_train_start(self, task_label):
        self.curr_train_task_label = task_label
        self.task_train_steps = 0
        self._reset_optimizers()
        return

    def task_train_end(self):
        self.curr_train_task_label = None
        return

    def task_eval_start(self, task_label):
        self.curr_eval_task_label = task_label
        self.network.eval()
        return

    def task_eval_end(self):
        self.curr_eval_task_label = None
        self.network.train()
        return


class SACLLAgent(SACContinualLearnerAgent):
    def __init__(self, config):
        SACContinualLearnerAgent.__init__(self, config)
        self.seen_tasks = {}
        self.new_task = False
        self.curr_train_task_label = None

    def _label_to_idx(self, task_label):
        eps = 1e-5
        found_task_idx = None
        for task_idx, seen_task_label in self.seen_tasks.items():
            if np.linalg.norm((task_label - seen_task_label), ord=2) < eps:
                found_task_idx = task_idx
                break
        return found_task_idx

    def _sync_target_for_task(self, task_idx, new_task=False):
        set_model_task(self.target_network, task_idx, new_task=new_task)
        self.target_network.load_state_dict(self.network.state_dict())

    def task_train_start(self, task_label):
        task_idx = self._label_to_idx(task_label)
        if task_idx is None:
            task_idx = len(self.seen_tasks)
            self.seen_tasks[task_idx] = task_label
            self.new_task = True
            set_model_task(self.network, task_idx, new_task=True)
        else:
            set_model_task(self.network, task_idx)
        self._sync_target_for_task(task_idx, new_task=self.new_task)
        self.curr_train_task_label = task_label
        self.task_train_steps = 0
        self._reset_optimizers()
        return

    def task_train_end(self):
        if self.new_task:
            consolidate_mask(self.network)
            cache_masks(self.network)
            set_num_tasks_learned(self.network, len(self.seen_tasks))
        else:
            cache_masks(self.network)

        self.target_network.load_state_dict(self.network.state_dict())
        cache_masks(self.target_network)
        set_num_tasks_learned(self.target_network, len(self.seen_tasks))

        self.curr_train_task_label = None
        self.new_task = False
        return

    def task_eval_start(self, task_label):
        self.network.eval()
        task_idx = self._label_to_idx(task_label)
        if task_idx is None:
            task_idx = 0
        set_model_task(self.network, task_idx)
        self.curr_eval_task_label = task_label
        return

    def task_eval_end(self):
        self.curr_eval_task_label = None
        self.network.train()
        if self.curr_train_task_label is not None:
            task_idx = self._label_to_idx(self.curr_train_task_label)
            set_model_task(self.network, task_idx)
            set_model_task(self.target_network, task_idx)
        return


class SACDetectLLAgent(SACLLAgent):
    def __init__(self, config):
        SACLLAgent.__init__(self, config)
        self.embedding_table = {}
        self._emb_indices = []
        self._emb_matrix = None
        self._emb_row = {}
        self.device = Config.DEVICE
        self.detect_input_dim = int(np.prod(self.task.state_dim))

        self.detect = config.detect_fn(self.detect_input_dim, self.task.action_dim)
        self.detect.set_reference(self.detect_input_dim, config.detect_reference_num,
            self.task.action_dim)

        self.task_emb_size = self.detect.precalculate_embedding_size(
            config.detect_reference_num,
            self.detect_input_dim,
            self.task.action_dim,
        )
        self.new_task_emb = None
        for task_idx, _ in enumerate(self.config.cl_tasks_info):
            self.embedding_table[task_idx] = None

    def extract_sar(self, batch_size=None):
        states, actions, rewards, _, _ = self.data_buffer.sample(batch_size=batch_size)

        if not isinstance(states, torch.Tensor):
            states = torch.as_tensor(states)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards)

        states = states.view(states.shape[0], -1)

        if isinstance(self.task.action_space, gym.spaces.Discrete):
            actions = actions.view(actions.shape[0], 1)
        else:
            actions = actions.view(actions.shape[0], self.task.action_dim)

        rewards = rewards.view(rewards.shape[0], 1)
        sar = torch.cat([states, actions, rewards], dim=1)
        return sar.detach()

    def compute_task_embedding(self, sar_data, action_space_size):
        with torch.no_grad():
            task_embedding = self.detect.lwe(sar_data, action_space_size)
        self.new_task_emb = task_embedding
        self.task_emb_size = len(task_embedding)
        return task_embedding

    def _update_embedding(self, task_idx: int, new_emb, ema: float = 0.5):
        if not isinstance(new_emb, torch.Tensor):
            new_emb = torch.as_tensor(new_emb, device=self.device, dtype=torch.float32)
        else:
            new_emb = new_emb.to(self.device, dtype=torch.float32)

        old = self.embedding_table[task_idx]
        if old is None:
            updated = new_emb
            is_new = True
        else:
            if not getattr(self.config, 'legacy_wte_ema', False):
                new_emb = F.normalize(new_emb, dim=0, eps=1e-8)
            updated = ema * old + (1 - ema) * new_emb
            is_new = False

        updated = F.normalize(updated, dim=0, eps=1e-8)
        self.embedding_table[task_idx] = updated

        if is_new:
            row = len(self._emb_indices)
            self._emb_indices.append(task_idx)
            self._emb_row[task_idx] = row
            if self._emb_matrix is None:
                self._emb_matrix = updated.unsqueeze(0)
            else:
                self._emb_matrix = torch.cat([self._emb_matrix, updated.unsqueeze(0)], dim=0)
        else:
            row = self._emb_row[task_idx]
            self._emb_matrix[row] = updated

    @torch.no_grad()
    def select_similar(self, task_idx: int, threshold: float = 0.5, topk=None):
        curr = self.embedding_table[task_idx]
        if curr is None or self._emb_matrix is None:
            return [], None

        sims = self._emb_matrix @ curr
        row = self._emb_row.get(task_idx, None)
        if row is not None:
            sims = sims.clone()
            sims[row] = -float('inf')

        mask = sims > threshold
        pos = torch.nonzero(mask, as_tuple=False).flatten()
        if pos.numel() == 0:
            return [], sims

        sel_sims = sims[pos]
        order = torch.argsort(sel_sims, descending=True)
        pos = pos[order]
        if topk is not None:
            pos = pos[:topk]

        selected = [self._emb_indices[i] for i in pos.tolist()]
        return selected, sims
