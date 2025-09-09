"""
Embedding layers for tokens and positions
"""
import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Token embedding layer with optional weight tying.
    
    Args:
        vocab_size: Size of the vocabulary
        embed_dim: Embedding dimension
        pad_token_id: ID of the padding token
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, pad_token_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pad_token_id = pad_token_id
        
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim))
        
        # Initialize padding token embedding to zeros
        with torch.no_grad():
            self.weight[pad_token_id].fill_(0)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            Token embeddings of shape (batch_size, seq_len, embed_dim)
        """
        return nn.functional.embedding(
            input_ids, self.weight, padding_idx=self.pad_token_id
        )


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings.
    
    Args:
        max_seq_len: Maximum sequence length
        embed_dim: Embedding dimension
    """
    
    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        self.weight = nn.Parameter(torch.randn(max_seq_len, embed_dim))
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Position embeddings of shape (seq_len, embed_dim)
        """
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        return self.weight[:seq_len]


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings as described in "Attention Is All You Need".
    These are fixed (non-learnable) embeddings.
    
    Args:
        max_seq_len: Maximum sequence length
        embed_dim: Embedding dimension
    """
    
    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # Create sinusoidal embeddings
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe, persistent=False)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Position embeddings of shape (seq_len, embed_dim)
        """
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        return self.pe[:seq_len]


class RelativePositionalEmbedding(nn.Module):
    """
    Relative positional embeddings as used in Transformer-XL and other models.
    Instead of absolute positions, this encodes relative distances between tokens.
    
    Args:
        embed_dim: Embedding dimension
        max_rel_dist: Maximum relative distance to consider
    """
    
    def __init__(self, embed_dim: int, max_rel_dist: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_rel_dist = max_rel_dist
        
        # Embeddings for relative distances
        # We need 2 * max_rel_dist + 1 embeddings (-max_rel_dist to +max_rel_dist)
        self.rel_embeddings = nn.Parameter(torch.randn(2 * max_rel_dist + 1, embed_dim))
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position embeddings of shape (seq_len, seq_len, embed_dim)
        """
        # Create relative position matrix
        positions = torch.arange(seq_len)
        rel_positions = positions[:, None] - positions[None, :]
        
        # Clip to max relative distance
        rel_positions = torch.clamp(
            rel_positions, -self.max_rel_dist, self.max_rel_dist
        )
        
        # Shift to positive indices
        rel_positions += self.max_rel_dist
        
        # Get embeddings
        rel_embeddings = self.rel_embeddings[rel_positions]
        
        return rel_embeddings


class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi) as described in the paper:
    "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
    
    This adds a linear bias to attention scores based on distance.
    
    Args:
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length for precomputed biases
    """
    
    def __init__(self, num_heads: int, max_seq_len: int = 8192):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Compute slopes for each attention head
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes, persistent=False)
        
        # Precompute biases for efficiency
        self._cached_biases = None
        self._cached_seq_len = 0
    
    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """Compute ALiBi slopes for each attention head."""
        
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # If num_heads is not a power of 2, we interpolate
            closest_power_of_2 = 2**math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes.extend(get_slopes_power_of_2(2*closest_power_of_2)[0::2][:num_heads-closest_power_of_2])
        
        return torch.tensor(slopes, dtype=torch.float32)
    
    def _build_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build ALiBi bias matrix for given sequence length."""
        # Create distance matrix
        positions = torch.arange(seq_len, device=device)
        distances = positions[:, None] - positions[None, :]
        distances = distances.abs()
        
        # Apply slopes to distances
        biases = distances[None, :, :] * self.slopes[:, None, None]
        biases = -biases  # Negative because we want closer tokens to have higher scores
        
        return biases
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            seq_len: Sequence length
            device: Device to place the bias tensor on
            
        Returns:
            ALiBi bias tensor of shape (num_heads, seq_len, seq_len)
        """
        if seq_len != self._cached_seq_len or self._cached_biases is None:
            self._cached_biases = self._build_alibi_bias(seq_len, device)
            self._cached_seq_len = seq_len
        
        return self._cached_biases
