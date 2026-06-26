#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from copy import deepcopy

from .network_utils import *
from .network_bodies import *

class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        y = self.fc_head(phi)
        if to_numpy:
            y = y.cpu().detach().numpy()
        return y

class VanillaNet_CL(nn.Module, BaseNet):
    def __init__(self, output_dim, task_label_dim, body):
        super(VanillaNet_CL, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.task_label_dim = task_label_dim
        self.to(Config.DEVICE)

    def predict(self, x, task_label=None, to_numpy=False):
        x = tensor(x)
        task_label = tensor(task_label)
        phi = self.body(x, task_label)
        y = self.fc_head(phi)
        if to_numpy:
            y = y.cpu().detach().numpy()
        return y

class DuelingNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        if to_numpy:
            return q.cpu().detach().numpy()
        return q

class DuelingNet_CL(nn.Module, BaseNet):
    def __init__(self, action_dim, task_label_dim, body):
        super(DuelingNet_CL, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.task_label_dim = task_label_dim
        self.to(Config.DEVICE)

    def predict(self, x, task_label=None, to_numpy=False):
        x = tensor(x)
        task_label = tensor(task_label)
        phi = self.body(x, task_label)
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        if to_numpy:
            return q.cpu().detach().numpy()
        return q

class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        if to_numpy:
            return prob.cpu().detach().numpy()
        return prob

class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        if to_numpy:
            quantiles = quantiles.cpu().detach().numpy()
        return quantiles

class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x):
        phi = self.body(tensor(x))
        q = self.fc_q(phi)
        beta = F.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        return q, beta, log_pi

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body):
        super(ActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())

class ActorCriticNetSS(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body, num_tasks, \
        new_task_mask, discrete_mask=True):
        super(ActorCriticNetSS, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = MultitaskMaskLinear(actor_body.feature_dim, action_dim, \
            discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)
        self.fc_critic = MultitaskMaskLinear(critic_body.feature_dim, 1, \
            discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)

        ap = [p for p in self.actor_body.parameters() if p.requires_grad is True]
        ap += [p for p in self.fc_action.parameters() if p.requires_grad is True]
        self.actor_params = ap

        cp = [p for p in self.critic_body.parameters() if p.requires_grad is True]
        cp += [p for p in self.fc_critic.parameters() if p.requires_grad is True]
        self.critic_params = cp

        self.phi_params = [p for p in self.phi_body.parameters() if p.requires_grad is True]

class DeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(DeterministicActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.actor_opt = actor_opt_fn(self.network.actor_params + self.network.phi_params)
        self.critic_opt = critic_opt_fn(self.network.critic_params + self.network.phi_params)
        self.to(Config.DEVICE)

    def predict(self, obs, to_numpy=False):
        phi = self.feature(obs)
        action = self.actor(phi)
        if to_numpy:
            return action.cpu().detach().numpy()
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.network.phi_body(obs)

    def actor(self, phi):
        return F.tanh(self.network.fc_action(self.network.actor_body(phi)))

    def critic(self, phi, a):
        return self.network.fc_critic(self.network.critic_body(phi, a))

class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.std = nn.Parameter(torch.ones(1, action_dim))
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, to_numpy=False):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        mean = F.tanh(self.network.fc_action(phi_a))
        if to_numpy:
            return mean.cpu().detach().numpy()
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, self.std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob, tensor(np.zeros((log_prob.size(0), 1))), v

# actor-critic net for continual learning where tasks are labelled using
# supermask superposition algorithm
class GaussianActorCriticNet_SS(nn.Module, BaseNet):
    LOG_STD_MIN = -0.6931 #-20.
    LOG_STD_MAX = 0.4055 #1.3
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 num_tasks=3,
                 new_task_mask='random'):
        super(GaussianActorCriticNet_SS, self).__init__()
        # continuous values mask is used for Gaussian (continuous control policies)
        discrete_mask = False
        self.network = ActorCriticNetSS(state_dim, action_dim, phi_body, actor_body, critic_body, \
            num_tasks, new_task_mask, discrete_mask=discrete_mask)
        self.task_label_dim = task_label_dim

        self.network.fc_log_std = MultitaskMaskLinear(self.network.actor_body.feature_dim, \
            action_dim, discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)
        self.network.actor_params += [p for p in self.network.fc_log_std.parameters() if p.requires_grad is True]
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, task_label=None, return_layer_output=False, to_numpy=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out
        #mean = F.tanh(self.network.fc_action(phi_a))
        mean = self.network.fc_action(phi_a)
        if to_numpy:
            return mean.cpu().detach().numpy()
        v = self.network.fc_critic(phi_v)
        log_std = self.network.fc_log_std(phi_a)
        log_std = torch.clamp(log_std, GaussianActorCriticNet_SS.LOG_STD_MIN, \
            GaussianActorCriticNet_SS.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_mean', mean), ('policy_std', std), \
                ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        entropy = dist.entropy()
        entropy = entropy.sum(-1).unsqueeze(-1)
        return mean, action, log_prob, entropy, v, layers_output

# actor-critic net for continual learning where tasks are labelled
class GaussianActorCriticNet_CL(nn.Module, BaseNet):
    LOG_STD_MIN = -0.6931 #-20.
    LOG_STD_MAX = 0.4055 #1.3
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet_CL, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.task_label_dim = task_label_dim

        self.network.fc_log_std = layer_init(nn.Linear(self.network.actor_body.feature_dim, \
            action_dim), 1e-3)
        self.network.actor_params += [p for p in self.network.fc_log_std.parameters() if p.requires_grad is True]
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, task_label=None, return_layer_output=False, to_numpy=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out
        #mean = F.tanh(self.network.fc_action(phi_a))
        mean = self.network.fc_action(phi_a)
        if to_numpy:
            return mean.cpu().detach().numpy()
        v = self.network.fc_critic(phi_v)
        log_std = self.network.fc_log_std(phi_a)
        log_std = torch.clamp(log_std, GaussianActorCriticNet_CL.LOG_STD_MIN, \
            GaussianActorCriticNet_CL.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_mean', mean), ('policy_std', std), \
                ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        entropy = dist.entropy()
        entropy = entropy.sum(-1).unsqueeze(-1)
        return mean, action, log_prob, entropy, v, layers_output

class _SACNetworkBase(nn.Module, BaseNet):
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0
    EPS = 1e-6

    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(_SACNetworkBase, self).__init__()
        if phi_body is None:
            phi_body = DummyBody_CL(state_dim, task_label_dim=task_label_dim)
        if actor_body is None:
            actor_body = FCBody_CL(phi_body.feature_dim, hidden_units=(256, 256), gate=F.relu)
        if critic_body is None:
            critic_body = FCBody_CL(phi_body.feature_dim + action_dim, hidden_units=(256, 256),
                gate=F.relu)

        self.action_dim = action_dim
        self.task_label_dim = task_label_dim
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic1_body = critic_body
        self.critic2_body = deepcopy(critic_body)

    def _actor_features(self, obs, task_label=None, return_layer_output=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.phi_body(obs, task_label, return_layer_output, 'phi_body')
        layers_output += out
        phi_a, out = self.actor_body(phi, None, return_layer_output, 'actor_body')
        layers_output += out
        return phi, phi_a, layers_output

    def _critic_features(self, obs, action, task_label=None, return_layer_output=False):
        obs = tensor(obs)
        action = tensor(action)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.phi_body(obs, task_label, return_layer_output, 'phi_body')
        layers_output += out
        q_input = torch.cat([phi, action], dim=1)
        phi_q1, out = self.critic1_body(q_input, None, return_layer_output, 'critic1_body')
        layers_output += out
        phi_q2, out = self.critic2_body(q_input, None, return_layer_output, 'critic2_body')
        layers_output += out
        return phi_q1, phi_q2, layers_output

    def _inverse_squash(self, action):
        action = action.clamp(-1.0 + self.EPS, 1.0 - self.EPS)
        return 0.5 * (torch.log1p(action) - torch.log1p(-action))

    def q(self, obs, action, task_label=None, return_layer_output=False):
        phi_q1, phi_q2, layers_output = self._critic_features(obs, action, task_label,
            return_layer_output)
        q1 = self.fc_q1(phi_q1)
        q2 = self.fc_q2(phi_q2)
        if return_layer_output:
            layers_output += [('critic_q1', q1), ('critic_q2', q2)]
        return q1, q2, layers_output

    def sample(self, obs, task_label=None, deterministic=False, return_layer_output=False,
               with_logprob=True):
        _, phi_a, layers_output = self._actor_features(obs, task_label, return_layer_output)
        mean = self.fc_action(phi_a)
        log_std = self.fc_log_std(phi_a)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        pre_tanh = mean if deterministic else dist.rsample()
        action = torch.tanh(pre_tanh)

        log_prob = None
        if with_logprob:
            log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + self.EPS)
            log_prob = log_prob.sum(dim=1, keepdim=True)

        entropy = dist.entropy().sum(dim=1, keepdim=True)
        if return_layer_output:
            layers_output += [('policy_mean', mean), ('policy_log_std', log_std),
                ('policy_action', action)]
        return action, log_prob, entropy, mean, log_std, layers_output

    def log_prob(self, obs, action, task_label=None):
        _, phi_a, _ = self._actor_features(obs, task_label, return_layer_output=False)
        mean = self.fc_action(phi_a)
        log_std = self.fc_log_std(phi_a)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        pre_tanh = self._inverse_squash(tensor(action))
        action = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + self.EPS)
        return log_prob.sum(dim=1, keepdim=True)

    def predict(self, obs, action=None, task_label=None, return_layer_output=False, to_numpy=False,
                deterministic=False):
        sampled_action, log_prob, entropy, mean, _, layers_output = self.sample(
            obs,
            task_label=task_label,
            deterministic=deterministic,
            return_layer_output=return_layer_output,
            with_logprob=True,
        )
        policy_action = torch.tanh(mean)
        if to_numpy:
            return policy_action.cpu().detach().numpy()

        q1, q2, q_layers = self.q(obs, sampled_action, task_label=task_label,
            return_layer_output=return_layer_output)
        if return_layer_output:
            layers_output += q_layers
        value = torch.min(q1, q2)
        return policy_action, sampled_action, log_prob, entropy, value, layers_output

class SACTanhGaussianNet_SS(_SACNetworkBase):
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 num_tasks=3,
                 new_task_mask='random'):
        super(SACTanhGaussianNet_SS, self).__init__(state_dim, action_dim, task_label_dim,
            phi_body, actor_body, critic_body)
        discrete_mask = False
        self.fc_action = MultitaskMaskLinear(self.actor_body.feature_dim, action_dim,
            discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)
        self.fc_log_std = MultitaskMaskLinear(self.actor_body.feature_dim, action_dim,
            discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)
        self.fc_q1 = MultitaskMaskLinear(self.critic1_body.feature_dim, 1,
            discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)
        self.fc_q2 = MultitaskMaskLinear(self.critic2_body.feature_dim, 1,
            discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)
        self.to(Config.DEVICE)

class SACTanhGaussianNet_CL(_SACNetworkBase):
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(SACTanhGaussianNet_CL, self).__init__(state_dim, action_dim, task_label_dim,
            phi_body, actor_body, critic_body)
        self.fc_action = layer_init(nn.Linear(self.actor_body.feature_dim, action_dim), 1e-3)
        self.fc_log_std = layer_init(nn.Linear(self.actor_body.feature_dim, action_dim), 1e-3)
        self.fc_q1 = layer_init(nn.Linear(self.critic1_body.feature_dim, 1), 1e-3)
        self.fc_q2 = layer_init(nn.Linear(self.critic2_body.feature_dim, 1), 1e-3)
        self.to(Config.DEVICE)

class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.to(Config.DEVICE)

    def predict(self, obs, action=None):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob, dist.entropy().unsqueeze(-1), v

# actor-critic net for continual learning where tasks are labelled using
# supermask superposition algorithm
class CategoricalActorCriticNet_SS(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 num_tasks=3,
                 new_task_mask='random'):
        super(CategoricalActorCriticNet_SS, self).__init__()
        self.network = ActorCriticNetSS(state_dim, action_dim, phi_body, actor_body, critic_body, num_tasks, new_task_mask)
        self.task_label_dim = task_label_dim
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, task_label=None, return_layer_output=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out

        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_logits', logits), ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return logits, action, log_prob, dist.entropy().unsqueeze(-1), v, layers_output

# actor-critic net for continual learning where tasks are labelled
class CategoricalActorCriticNet_CL(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet_CL, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.task_label_dim = task_label_dim
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, task_label=None, return_layer_output=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out

        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_logits', logits), ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return logits, action, log_prob, dist.entropy().unsqueeze(-1), v, layers_output
