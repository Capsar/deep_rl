import torch as th

from src.deep_rl.deep_learning_models.feature_extractor import DLinearFeatureExtractor, GRUFeatureExtractor, LinearFeatureExtractor, TransformerFeatureExtractor
from src.deep_rl.replay_buffer import ReplayBuffer

class Critic(th.nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.state_layer = th.nn.Sequential(LinearFeatureExtractor(state_dim, hidden_dim), th.nn.LeakyReLU())
        self.action_layer = th.nn.Sequential(th.nn.Linear(action_dim, hidden_dim), th.nn.LeakyReLU())
        self.q_value = th.nn.Sequential(th.nn.Linear(hidden_dim * 2, hidden_dim), th.nn.LeakyReLU(),
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
