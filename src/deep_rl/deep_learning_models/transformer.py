import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Wide mult-head self-attention layer.

    Args:
        k: embedding dimension
        heads: number of heads (k mod heads must be 0)

    """

    def __init__(self, k, heads=8):
        super(MultiHeadAttention, self).__init__()

        self.heads = heads

        # These compute the queries, keys and values for all
        # heads (as a single concatenated vector)
        self.to_keys = nn.Linear(k, k * heads, bias=False)
        self.to_queries = nn.Linear(k, k * heads, bias=False)
        self.to_values = nn.Linear(k, k * heads, bias=False)

        # This unifies the outputs of the different heads into a single k-vector
        self.unify_heads = nn.Linear(k * heads, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        # Project input to queries, keys and values
        queries = self.to_queries(x).view(b, t, h, k)
        keys = self.to_keys(x).view(b, t, h, k)
        values = self.to_values(x).view(b, t, h, k)

        # Fold heads into the batch dimension
        keys = keys.transpose(1, 2).reshape(b * h, t, k)
        queries = queries.transpose(1, 2).reshape(b * h, t, k)
        values = values.transpose(1, 2).reshape(b * h, t, k)

        # Compute attention weights
        w_prime = torch.bmm(queries, keys.transpose(1, 2))
        w_prime = w_prime / (k ** (1 / 2))
        w = F.softmax(w_prime, dim=2)

        # Apply the self-attention to the values
        y = torch.bmm(w, values).view(b, h, t, k)

        # Swap h, t back, unify heads
        y = y.transpose(1, 2).reshape(b, t, h * k)

        y = self.unify_heads(y)

        return y


class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        """
        Basic transformer block.

        Args:
            k: embedding dimension
            heads: number of heads (k mod heads must be 0)

        """
        super(TransformerBlock, self).__init__()

        self.att = MultiHeadAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )

        self.norm2 = nn.LayerNorm(k)

    def forward(self, x):
        """
        Forward pass of trasformer block.

        Args:
            x: input with shape of (b, k)

        Returns:
            y: output with shape of (b, k)
        """
        # Self-attend
        y = self.att(x)

        # First residual connection
        x = x + y

        # Normalize
        x = self.norm1(x)

        # Pass through feed-forward network
        y = self.ff(x)

        # Second residual connection
        x = x + y

        # Again normalize
        y = self.norm2(x)

        return y


class Transformer(nn.Module):
    def __init__(self, k, heads=8, num_layers=2, input_length=40, num_inputs=256, num_outputs=10):
        """
        Transformer architecture.

        Args:
            k: embedding dimension
            heads: number of attention heads
            num_layers: number of transformer blocks in network
            input_length: length of input sequence
            num_inputs: input dimension
            num_outputs: ouput dimension
        """
        super(Transformer, self).__init__()

        # Embedding layers for input and position
        self.input_embedding = nn.Embedding(num_inputs, k)
        self.position_embedding = nn.Embedding(input_length, k)

        # Create transformer blocks
        blocks = [TransformerBlock(k, heads) for _ in range(num_layers)]
        self.blocks = nn.Sequential(*blocks)

        # Projects the output to desired output size
        self.output_projector = nn.Linear(k, num_outputs)

    def forward(self, x):
        """
        Forward pass of transformer model.

        Args:
            x: input with shape of (b, t)
        """
        should_squeeze = False
        if len(x.shape) == 2:
            should_squeeze = True
            x = x.unsqueeze(0)  # For single input batch first

        b, t, _ = x.shape

        # Add positional embedding
        p = torch.arange(t, device=x.device).view(1, t).expand(b, t)
        p = self.position_embedding(p)
        x = x + p

        # Compute transformer output
        x = self.blocks(x)

        # Average-pool over dimension t
        x = x.mean(dim=1)

        # Project output to desired size
        x = self.output_projector(x)

        if should_squeeze:
            x = x.squeeze(0)

        return x
