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

import torch as th
import torch.nn.functional as F

from src.deep_rl.deep_learning_models.actor import ContinuousActor, DLinearActor, GRUActor, TransformerActor
from src.deep_rl.deep_learning_models.critic import Critic, DLinearCritic, GRUCritic, MemoryDuelingNetworkCritic, TransformerCritic

def reset_model_weights(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
    else:
        if hasattr(layer, 'children'):
            for child in layer.children():
                reset_model_weights(child)


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
