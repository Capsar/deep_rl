import torch as th
from src.deep_rl.deep_learning_models.dlinear import DLinear

from src.deep_rl.deep_learning_models.transformer import Transformer

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