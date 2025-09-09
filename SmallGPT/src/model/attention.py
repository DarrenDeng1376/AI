"""
Multi-head attention mechanism implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        bias: Whether to use bias in linear layers
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(1, 1, 1024, 1024)).bool(),
            persistent=False
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply causal mask (for autoregressive generation)
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention mask to match scores shape
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
            scores = scores.masked_fill(~attention_mask.bool(), float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        out = out.reshape(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)
        out = self.output_proj(out)
        
        return out


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention (single head).
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            q: Query tensor of shape (..., seq_len, d_k)
            k: Key tensor of shape (..., seq_len, d_k)
            v: Value tensor of shape (..., seq_len, d_v)
            mask: Mask tensor of shape (..., seq_len, seq_len)
            
        Returns:
            Output tensor of shape (..., seq_len, d_v)
        """
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        return output


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) implementation.
    Alternative to standard positional embeddings.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Cache for rotary embeddings
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cos and sin values."""
        if seq_len > self._cached_seq_len or self._cached_cos is None:
            self._cached_seq_len = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cached_cos = emb.cos()
            self._cached_sin = emb.sin()
    
    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor
            seq_dim: Sequence dimension
            
        Returns:
            Tuple of (cos, sin) embeddings
        """
        seq_len = x.shape[seq_dim]
        self._update_cache(seq_len, x.device, x.dtype)
        
        return (
            self._cached_cos[:seq_len].to(dtype=x.dtype),
            self._cached_sin[:seq_len].to(dtype=x.dtype)
        )


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary positional embedding to query and key tensors."""
    
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed
