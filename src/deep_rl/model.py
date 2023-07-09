"""
Actor Critic Model
Critic:
- 2 Critics (Pessimistic)
- Dueling Architecture (for dealing with similar rewards for actions)

Actor:
- Stochastic Guassian (for dealing with continuous actions)
- Shared parameters before separate mean & std layers.

"""
from typing import List

import torch
import torch as th
import torch.nn.functional as F

from src.deep_rl.deep_learning_models import DLinear
from src.deep_rl.replay_buffer import ReplayBuffer
from src.deep_rl.deep_learning_models.transformer import Transformer


def reset_model_weights(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
    else:
        if hasattr(layer, 'children'):
            for child in layer.children():
                reset_model_weights(child)


class LinearFeatureExtractor(th.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearFeatureExtractor, self).__init__()
        self.linear = th.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class GRUFeatureExtractor(th.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUFeatureExtractor, self).__init__()
        self.gru = th.nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, hn = self.gru(x)

        return hn.squeeze(0)


class TransformerFeatureExtractor(th.nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim):
        super(TransformerFeatureExtractor, self).__init__()
        self.transformer = Transformer(k=input_dim, input_length=seq_len, num_inputs=input_dim, num_outputs=hidden_dim)

    def forward(self, x):
        return self.transformer(x)


class DLinearFeatureExtractor(th.nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim):
        super(DLinearFeatureExtractor, self).__init__()
        self.d_linear = DLinear(seq_len, 1, input_dim)
        self.linear = th.nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = self.d_linear(x)
        x = self.linear(x)
        if len(x.shape) > 2:
            x = x.permute(1, 0, 2)

        return x.squeeze(0)


class Critic(th.nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.state_layer = th.nn.Sequential(LinearFeatureExtractor(state_dim, hidden_dim), th.nn.LeakyReLU())
        self.action_layer = th.nn.Sequential(th.nn.Linear(action_dim, hidden_dim), th.nn.LeakyReLU())
        self.q_value = th.nn.Sequential(th.nn.Linear(hidden_dim * 2, hidden_dim), th.nn.LeakyReLU(),
                                        th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                        th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                        th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                        th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                        th.nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        h1 = self.state_layer(state)
        h2 = self.action_layer(action)
        return self.q_value(th.cat([h1, h2], dim=1))


class MemoryDuelingNetworkCritic(th.nn.Module):

    def __init__(self, input_dim, action_dim, hidden_dim, advantage_n=50):
        super(MemoryDuelingNetworkCritic, self).__init__()
        self.state_layer = th.nn.Sequential(th.nn.Linear(input_dim, hidden_dim), th.nn.LeakyReLU())
        self.action_layer = th.nn.Sequential(th.nn.Linear(action_dim, hidden_dim), th.nn.LeakyReLU())
        self.base = th.nn.Sequential(th.nn.Linear(hidden_dim * 2, hidden_dim), th.nn.LeakyReLU(),
                                     th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU())
        self.value = th.nn.Sequential(th.nn.Linear(hidden_dim, 1))
        self.advantage = th.nn.Sequential(th.nn.Linear(hidden_dim, 1))
        self.advantage_n = advantage_n
        self.memory_shape, self.action_memory = None, None

    def forward(self, x, action):
        if self.action_memory is None:
            self.memory_shape = {'actions': action.shape}
            self.action_memory = ReplayBuffer(200, self.memory_shape, batch_size=0)
        self.action_memory.store(actions=action)
        sampled_actions = self.action_memory.sample(self.advantage_n)['actions']
        h1 = self.state_layer(x).expand(sampled_actions.shape[0] + 1, -1, -1)
        action = th.cat([action.unsqueeze(dim=0), sampled_actions], dim=0)
        extracted_features = self.base(th.cat([h1, self.action_layer(action)], dim=2))
        value, advantages = self.value(extracted_features[0]), self.advantage(extracted_features)
        return value + advantages[0] - advantages.mean(dim=0)


# class PartitionDuelingNetworkCritic(th.nn.Module):
#
#     def __init__(self, input_dim, action_dim, hidden_dim, action_range, advantage_n=10):
#         super(PartitionDuelingNetworkCritic, self).__init__()
#         self.state_layer = th.nn.Sequential(th.nn.Linear(input_dim, hidden_dim), th.nn.LeakyReLU())
#         self.action_layer = th.nn.Sequential(th.nn.Linear(action_dim, hidden_dim), th.nn.LeakyReLU())
#         self.base = th.nn.Sequential(th.nn.Linear(hidden_dim * 2, hidden_dim), th.nn.LeakyReLU(),
#                                      th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU())
#         self.value = th.nn.Sequential(th.nn.Linear(hidden_dim, 1))
#         self.advantage = th.nn.Sequential(th.nn.Linear(hidden_dim, 1))
#         self.advantage_n = advantage_n
#         self.action_range = action_range
#         self.action_partitions = th.linspace(self.action_range[0], self.action_range[1], self.advantage_n)
#
#     def forward(self, x, action):
#         h1 = self.state_layer(x).expand(sampled_actions.shape[0]+1, -1, -1)
#         action = th.cat([action.unsqueeze(dim=0), sampled_actions], dim=0)
#         extracted_features = self.base(th.cat([h1, self.action_layer(action)], dim=2))
#         value, advantages = self.value(extracted_features[0]), self.advantage(extracted_features)
#         return value + advantages[0] - advantages[1:].mean(dim=0)

class GRUCritic(Critic):

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(GRUCritic, self).__init__(state_dim, action_dim, hidden_dim)
        self.state_layer = th.nn.Sequential(GRUFeatureExtractor(state_dim, hidden_dim), th.nn.LeakyReLU())


class TransformerCritic(Critic):
    def __init__(self, seq_len, state_dim, action_dim, hidden_dim):
        super(TransformerCritic, self).__init__(state_dim, action_dim, hidden_dim)
        self.state_layer = th.nn.Sequential(TransformerFeatureExtractor(seq_len, state_dim, hidden_dim), th.nn.LeakyReLU())


class DLinearCritic(Critic):
    def __init__(self, seq_len, state_dim, action_dim, hidden_dim):
        super(DLinearCritic, self).__init__(state_dim, action_dim, hidden_dim)
        self.state_layer = th.nn.Sequential(DLinearFeatureExtractor(seq_len, state_dim, hidden_dim), th.nn.LeakyReLU())


class ContinuousActor(th.nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, action_range, log_std_range=(-20, 2)):
        super(ContinuousActor, self).__init__()
        self.actions_base = th.nn.Sequential(LinearFeatureExtractor(input_dim, hidden_dim), th.nn.LeakyReLU(),
                                            th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                            th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                            th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                            th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                            th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                            th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU())
        self.actions_mean = th.nn.Linear(hidden_dim, output_dim)
        self.actions_log_std = th.nn.Linear(hidden_dim, output_dim)

        self.ACTION_MIN = action_range[0]
        self.ACTION_MAX = action_range[1]
        self.ACTION_SCALE = 0.5 * (self.ACTION_MAX - self.ACTION_MIN)
        self.ACTION_BIAS = 0.5 * (self.ACTION_MAX + self.ACTION_MIN)

        self.LOG_STD_MIN = log_std_range[0]
        self.LOG_STD_MAX = log_std_range[1]

    def forward(self, x):
        x = self.actions_base(x)
        actions_std = th.clamp(self.actions_log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
        return th.distributions.Normal(self.actions_mean(x), actions_std)

    def sample_log_prob(self, x):
        dist = self.forward(x)
        x = dist.rsample()
        y = th.tanh(x)
        action = self.ACTION_SCALE * y + self.ACTION_BIAS
        log_prob = dist.log_prob(x) - th.log(self.ACTION_SCALE * (1 - y.pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob

    def act(self, x):
        with torch.no_grad():
            return self.sample_log_prob(x)[0]

    def get_deterministic_action(self, x):
        with torch.no_grad():
            x = self.actions_base(x)
            x = th.tanh(self.actions_mean(x))
            return x * self.ACTION_SCALE + self.ACTION_BIAS


class GRUActor(ContinuousActor):

    def __init__(self, input_dim, output_dim, hidden_dim, action_range, log_std_range=(-20, 2)):
        super(GRUActor, self).__init__(input_dim, output_dim, hidden_dim, action_range, log_std_range)
        self.actions_base = th.nn.Sequential(GRUFeatureExtractor(input_dim, hidden_dim), th.nn.LeakyReLU(),
                                             th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                             th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                             th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                             th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                             th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                             th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU())


class TransformerActor(ContinuousActor):
    def __init__(self, seq_len, input_dim, output_dim, hidden_dim, action_range, log_std_range=(-20, 2)):
        super(TransformerActor, self).__init__(input_dim, output_dim, hidden_dim, action_range, log_std_range)
        self.actions_base = th.nn.Sequential(TransformerFeatureExtractor(seq_len, input_dim, hidden_dim), th.nn.LeakyReLU(),
                                             th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                             th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU())


class DLinearActor(ContinuousActor):
    def __init__(self, seq_len, input_dim, output_dim, hidden_dim, action_range, log_std_range=(-20, 2)):
        super(DLinearActor, self).__init__(input_dim, output_dim, hidden_dim, action_range, log_std_range)
        self.actions_base = th.nn.Sequential(DLinearFeatureExtractor(seq_len, input_dim, hidden_dim), th.nn.LeakyReLU(),
                                             th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                             th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU())


class SoftActorCritic(th.nn.Module):

    def __init__(self, obv_space, action_space, hidden_dim: List[int], actor_class=ContinuousActor, critic_class=Critic, init_classes=True):
        super(SoftActorCritic, self).__init__()
        if init_classes:
            state_dim = obv_space.shape[-1]
            action_dim = action_space.shape[-1]
            action_range = (action_space.low[0], action_space.high[0])
            self.actor = actor_class(state_dim, action_dim, hidden_dim[0], action_range)
            self.critic1 = critic_class(state_dim, action_dim, hidden_dim[1])  # Will still provide 1 value. action_dim is used for input too.
            self.critic2 = critic_class(state_dim, action_dim, hidden_dim[1])
            self.t_critic1 = critic_class(state_dim, action_dim, hidden_dim[1])  # Will still provide 1 value. action_dim is used for input too.
            self.t_critic2 = critic_class(state_dim, action_dim, hidden_dim[1])
            for p in self.t_critic1.parameters():
                p.requires_grad = False
            for p in self.t_critic2.parameters():
                p.requires_grad = False
            self.t_critic1.load_state_dict(self.critic1.state_dict())
            self.t_critic2.load_state_dict(self.critic2.state_dict())

    def act(self, x):
        return self.actor.act(x)

    def sample_log_prob(self, x):
        return self.actor.sample_log_prob(x)

    def get_deterministic_action(self, x):
        return self.actor.get_deterministic_action(x)

    def q_values(self, x, action, target=False):
        if target:
            return self.t_critic1(x, action), self.t_critic2(x, action)
        return self.critic1(x, action), self.critic2(x, action)


class DuelingSoftActorCritic(SoftActorCritic):
    def __init__(self, obv_space, action_space, hidden_dim: List[int], action_range):
        super(DuelingSoftActorCritic, self).__init__(obv_space, action_space, hidden_dim, action_range, critic_class=MemoryDuelingNetworkCritic)


class GRUSoftActorCritic(SoftActorCritic):
    def __init__(self, obv_space, action_space, hidden_dim: List[int]):
        super(GRUSoftActorCritic, self).__init__(obv_space, action_space, hidden_dim, actor_class=GRUActor, critic_class=GRUCritic)


class DLinearSoftActorCritic(SoftActorCritic):
    def __init__(self, obv_space, action_space, hidden_dim: List[int], actor_class=DLinearActor, critic_class=DLinearCritic):
        super().__init__(obv_space, action_space, hidden_dim, actor_class, critic_class, False)
        state_seq_len = obv_space.shape[0]
        state_dim = obv_space.shape[-1]
        action_dim = action_space.shape[-1]
        action_range = (action_space.low[0], action_space.high[0])

        self.actor = actor_class(state_seq_len, state_dim, action_dim, hidden_dim[0], action_range)
        self.critic1 = critic_class(state_seq_len, state_dim, action_dim, hidden_dim[1])  # Will still provide 1 value. action_dim is used for input too.
        self.critic2 = critic_class(state_seq_len, state_dim, action_dim, hidden_dim[1])
        self.t_critic1 = critic_class(state_seq_len, state_dim, action_dim, hidden_dim[1])  # Will still provide 1 value. action_dim is used for input too.
        self.t_critic2 = critic_class(state_seq_len, state_dim, action_dim, hidden_dim[1])
        for p in self.t_critic1.parameters():
            p.requires_grad = False
        for p in self.t_critic2.parameters():
            p.requires_grad = False
        self.t_critic1.load_state_dict(self.critic1.state_dict())
        self.t_critic2.load_state_dict(self.critic2.state_dict())


class TransformerSoftActorCritic(SoftActorCritic):
    def __init__(self, obv_space, action_space, hidden_dim: List[int], actor_class=TransformerActor, critic_class=TransformerCritic):
        super().__init__(obv_space, action_space, hidden_dim, actor_class, critic_class, False)
        state_seq_len = obv_space.shape[0]
        state_dim = obv_space.shape[-1]
        action_dim = action_space.shape[-1]
        action_range = (action_space.low[0], action_space.high[0])

        self.actor = actor_class(state_seq_len, state_dim, action_dim, hidden_dim[0], action_range)
        self.critic1 = critic_class(state_seq_len, state_dim, action_dim, hidden_dim[1])  # Will still provide 1 value. action_dim is used for input too.
        self.critic2 = critic_class(state_seq_len, state_dim, action_dim, hidden_dim[1])
        self.t_critic1 = critic_class(state_seq_len, state_dim, action_dim, hidden_dim[1])  # Will still provide 1 value. action_dim is used for input too.
        self.t_critic2 = critic_class(state_seq_len, state_dim, action_dim, hidden_dim[1])
        for p in self.t_critic1.parameters():
            p.requires_grad = False
        for p in self.t_critic2.parameters():
            p.requires_grad = False
        self.t_critic1.load_state_dict(self.critic1.state_dict())
        self.t_critic2.load_state_dict(self.critic2.state_dict())
