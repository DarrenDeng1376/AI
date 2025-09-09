"""
SmallGPT: A minimal GPT implementation
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .attention import MultiHeadAttention
from .transformer import TransformerBlock
from .embeddings import TokenEmbedding, PositionalEmbedding


class SmallGPT(nn.Module):
    """
    A small GPT model implementation with configurable parameters.
    
    Args:
        vocab_size: Size of the vocabulary
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        pad_token_id: Padding token ID
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        
        # Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim, pad_token_id)
        self.position_embedding = PositionalEmbedding(max_seq_len, embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_final = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Tie weights between token embedding and output projection
        self.output_projection.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights using normal distribution with std=0.02"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            return_dict: Whether to return a dictionary
            
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Check sequence length
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        # Token and position embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(seq_len)
        
        # Combine embeddings
        x = self.embedding_dropout(token_embeds + position_embeds)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # Final layer norm and output projection
        x = self.ln_final(x)
        logits = self.output_projection(x)
        
        if return_dict:
            return {
                'logits': logits,
                'hidden_states': x
            }
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated token IDs of shape (batch_size, generated_length)
        """
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        
        self.eval()
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(generated)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Get logits for the last position
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply sampling strategies
                if do_sample:
                    next_token = self._sample_next_token(
                        next_token_logits, top_k=top_k, top_p=top_p
                    )
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append generated token
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end-of-sequence
                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
                
                # Check if we've reached max sequence length
                if generated.shape[1] >= self.max_seq_len:
                    break
        
        return generated
    
    def _sample_next_token(
        self, 
        logits: torch.Tensor, 
        top_k: Optional[int] = None, 
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Sample next token using top-k and/or top-p sampling."""
        
        # Top-k sampling
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(-1, top_k_indices, top_k_logits)
        
        # Top-p (nucleus) sampling
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        
        Args:
            non_embedding: If True, exclude embedding parameters from count
            
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.position_embedding.weight.numel()
        return n_params
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> 'SmallGPT':
        """Load a pretrained model from a checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Extract model configuration
        config = checkpoint.get('config', {})
        model = cls(**config, **kwargs)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def save_pretrained(self, path: str, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'max_seq_len': self.max_seq_len,
                'dropout': self.dropout,
                'pad_token_id': self.pad_token_id,
            },
            **kwargs
        }
        torch.save(checkpoint, path)
