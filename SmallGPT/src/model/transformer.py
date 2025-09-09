"""
Transformer block implementation
"""
import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    A single transformer block consisting of:
    1. Multi-head self-attention
    2. Feed-forward network
    3. Residual connections and layer normalization
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        ff_dim: Feed-forward dimension (if None, defaults to 4 * embed_dim)
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        dropout: float = 0.1,
        ff_dim: Optional[int] = None
    ):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = 4 * embed_dim
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with pre-norm architecture.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Self-attention with residual connection (pre-norm)
        norm_x = self.ln1(x)
        attn_out = self.attention(norm_x, attention_mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection (pre-norm)
        norm_x = self.ln2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)
        
        return x


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Args:
        embed_dim: Input embedding dimension
        ff_dim: Hidden dimension of the feed-forward network
        dropout: Dropout rate
        activation: Activation function ('relu', 'gelu', 'swish')
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        ff_dim: int, 
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()  # SiLU is the same as Swish
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class GLU(nn.Module):
    """
    Gated Linear Unit activation function.
    GLU(x) = (W1 * x + b1) ⊙ σ(W2 * x + b2)
    where ⊙ is element-wise multiplication and σ is sigmoid.
    """
    
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.linear_gate = nn.Linear(embed_dim, ff_dim)
        self.linear_value = nn.Linear(embed_dim, ff_dim)
        self.output_proj = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated linear unit.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        gate = torch.sigmoid(self.linear_gate(x))
        value = self.linear_value(x)
        x = gate * value
        x = self.dropout(x)
        x = self.output_proj(x)
        return x


class SwiGLU(nn.Module):
    """
    SwiGLU activation function: SwiGLU(x) = Swish(W1 * x) ⊙ (W2 * x)
    Used in some modern transformer variants like PaLM.
    """
    
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Note: We use ff_dim * 2 // 3 to keep parameter count similar to standard FFN
        hidden_dim = int(ff_dim * 2 / 3)
        
        self.linear_gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.linear_value = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.output_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        gate = torch.nn.functional.silu(self.linear_gate(x))  # Swish activation
        value = self.linear_value(x)
        x = gate * value
        x = self.dropout(x)
        x = self.output_proj(x)
        return x


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Alternative to LayerNorm that doesn't center the data.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
