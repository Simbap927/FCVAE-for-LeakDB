"""
Self-Attention Mechanism for Time Series
- Multi-Head Attention for capturing long-range dependencies
- Transformer-based encoder layer
- Python 3.11 compatible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    - Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        """
        Args:
            temperature: Scaling factor (usually sqrt(d_k))
            attn_dropout: Dropout rate for attention weights
        """
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query [batch * n_head, len_q, d_k]
            k: Key [batch * n_head, len_k, d_k]
            v: Value [batch * n_head, len_v, d_v]
            mask: Attention mask (optional)

        Returns:
            output: [batch * n_head, len_q, d_v]
            attn: Attention weights [batch * n_head, len_q, len_k]
        """
        # Compute attention scores: QK^T / sqrt(d_k)
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature

        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Softmax + Dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum: Attention * V
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    - Multiple attention heads capture different representation subspaces
    """

    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1
    ):
        """
        Args:
            n_head: Number of attention heads
            d_model: Model dimension
            d_k: Key/Query dimension per head
            d_v: Value dimension per head
            dropout: Dropout rate
        """
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # Linear projections for Q, K, V
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        # Output projection
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        # Scaled Dot-Product Attention
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # Dropout and LayerNorm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence"""
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (self.d_k + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (self.d_k + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (self.d_k + self.d_v)))
        nn.init.xavier_normal_(self.fc.weight)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query [batch, len_q, d_model]
            k: Key [batch, len_k, d_model]
            v: Value [batch, len_v, d_model]
            mask: Attention mask (optional)

        Returns:
            output: [batch, len_q, d_model]
            attn: Attention weights [batch, n_head, len_q, len_k]
        """
        batch_size, len_q, d_model = q.size()
        _, len_k, _ = k.size()
        _, len_v, _ = v.size()

        # Residual connection
        residual = q

        # Linear projections and split into multiple heads
        # [batch, len, d_model] -> [batch, len, n_head, d_k] -> [batch, n_head, len, d_k]
        q = self.w_qs(q).view(batch_size, len_q, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_ks(k).view(batch_size, len_k, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_vs(v).view(batch_size, len_v, self.n_head, self.d_v).transpose(1, 2)

        # Reshape for batch matrix multiplication
        # [batch, n_head, len, d_k] -> [batch * n_head, len, d_k]
        q = q.contiguous().view(-1, len_q, self.d_k)
        k = k.contiguous().view(-1, len_k, self.d_k)
        v = v.contiguous().view(-1, len_v, self.d_v)

        # Apply attention mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
            mask = mask.view(-1, len_q, len_k)

        # Scaled Dot-Product Attention
        output, attn = self.attention(q, k, v, mask=mask)

        # Reshape and concatenate heads
        # [batch * n_head, len_q, d_v] -> [batch, n_head, len_q, d_v] -> [batch, len_q, n_head * d_v]
        output = output.view(batch_size, self.n_head, len_q, self.d_v)
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, -1)

        # Output projection
        output = self.dropout(self.fc(output))

        # Residual connection + LayerNorm
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    - FFN(x) = max(0, xW1 + b1)W2 + b2
    - Applied to each position separately and identically
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        # Two 1D convolutions (equivalent to position-wise linear layers)
        self.w_1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.w_2 = nn.Conv1d(d_ff, d_model, kernel_size=1)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
        """
        residual = x

        # Transpose for Conv1d: [batch, d_model, seq_len]
        x = x.transpose(1, 2)

        # Feed-forward network
        x = self.w_2(F.relu(self.w_1(x)))

        # Transpose back: [batch, seq_len, d_model]
        x = x.transpose(1, 2)

        # Dropout
        x = self.dropout(x)

        # Residual connection + LayerNorm
        output = self.layer_norm(x + residual)

        return output


class SelfAttentionLayer(nn.Module):
    """
    Self-Attention Encoder Layer
    - Multi-Head Self-Attention + Position-wise FFN
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_head: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension
            n_head: Number of attention heads
            d_k: Key/Query dimension per head
            d_v: Value dimension per head
            dropout: Dropout rate
        """
        super().__init__()

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: Attention mask (optional)

        Returns:
            output: [batch, seq_len, d_model]
            attn: Attention weights
        """
        # Self-Attention (Q=K=V=x)
        output, attn = self.slf_attn(x, x, x, mask=mask)

        # Position-wise Feed-Forward
        output = self.pos_ffn(output)

        return output, attn


# ===== Test =====
if __name__ == '__main__':
    print("=" * 70)
    print("Testing Self-Attention Modules")
    print("=" * 70)

    # Hyperparameters
    batch_size = 32
    seq_len = 96
    d_model = 256
    d_ff = 512
    n_head = 8
    d_k = d_model // n_head
    d_v = d_model // n_head

    # Random input
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {x.shape}")

    # Test ScaledDotProductAttention
    print("\n" + "-" * 70)
    print("Testing ScaledDotProductAttention")
    print("-" * 70)
    attn_module = ScaledDotProductAttention(temperature=d_k ** 0.5)
    q = k = v = x.view(-1, seq_len, d_k)  # Simplified for testing
    output, attn_weights = attn_module(q, k, v)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")

    # Test MultiHeadAttention
    print("\n" + "-" * 70)
    print("Testing MultiHeadAttention")
    print("-" * 70)
    mha = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=0.1)
    output, attn = mha(x, x, x)
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attn.shape}")

    # Test PositionwiseFeedForward
    print("\n" + "-" * 70)
    print("Testing PositionwiseFeedForward")
    print("-" * 70)
    ffn = PositionwiseFeedForward(d_model, d_ff, dropout=0.1)
    output = ffn(x)
    print(f"Output shape: {output.shape}")

    # Test SelfAttentionLayer
    print("\n" + "-" * 70)
    print("Testing SelfAttentionLayer")
    print("-" * 70)
    sa_layer = SelfAttentionLayer(d_model, d_ff, n_head, d_k, d_v, dropout=0.1)
    output, attn = sa_layer(x)
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attn.shape}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
